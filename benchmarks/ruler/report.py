"""Official RULER scoring, paired uncertainty, and bounded serving trade-offs."""
import ast
import importlib.util
import json
import re
from pathlib import Path
import sys
import numpy as np
from common import ROOT, UPSTREAM, WORKSPACE, ARMS, LENGTHS, read_jsonl, write_json, digest

spec=importlib.util.spec_from_file_location('ruler_metrics',UPSTREAM/'scripts/eval/synthetic/constants.py')
metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
tree=ast.parse((UPSTREAM/'scripts/eval/evaluate.py').read_text())
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='postprocess_pred')
scope={'re':re};exec(compile(ast.Module(body=[function],type_ignores=[]),'<official postprocess>','exec'),scope)

def score(prediction, references, category):
    cleaned=scope['postprocess_pred'](prediction,{})
    return metrics.TASKS[category]['metric_fn']([cleaned],[references])/100

def paired_interval(differences, seed=42):
    """Rows = tasks, columns = samples. Resample pairs within each task."""
    delta=np.asarray(differences,dtype=float)
    rng=np.random.default_rng(seed)
    indices=rng.integers(delta.shape[1],size=(10000,delta.shape[0],delta.shape[1]))
    draws=delta[np.arange(delta.shape[0])[None,:,None],indices].mean(axis=(1,2))*100
    return [float(x) for x in np.quantile(draws,[.025,.975])]

def main():
    selection=json.loads((ROOT/'formal-selection.json').read_text());n=selection['n']
    data={r['id']:r for path in (ROOT/'data').glob('*/*.jsonl') for r in read_jsonl(path) if r['sample']<n}
    tasks=sorted({r['task'] for r in data.values()})
    results={};comparisons={};all_deltas={};audit=[]
    for length in LENGTHS:
        results[str(length)]={};byarm={}
        for arm in ARMS:
            folder=ROOT/'runs'/f'formal-{length}-{arm}'
            summary=json.loads((folder/'summary.json').read_text())
            rows=read_jsonl(folder/'results.jsonl')
            assert len(rows)==len(tasks)*n and len({r['id'] for r in rows})==len(rows)
            mapping={r['id']:r for r in rows};byarm[arm]=mapping
            scores={task:[] for task in tasks}
            for r in rows:
                source=data[r['id']]
                r['score']=score(r['prediction'],source['outputs'],source['category'])
                scores[r['task']].append(r['score'])
                assert r['cached_tokens']==0 and r['cache_hit_delta']==0 and r['preemptions']==0
            task_scores={task:float(np.mean(values)*100) for task,values in scores.items()}
            total_tokens=sum(r['usage']['completion_tokens'] for r in rows)
            total_time=sum(r['e2e'] for r in rows)
            post=[r['post_first_rate'] for r in rows if r['post_first_rate'] is not None]
            results[str(length)][arm]={**summary,'score':float(np.mean(list(task_scores.values()))),'task_scores':task_scores,
                'fully_correct':sum(r['score']==1 for r in rows),'empty':sum(not r['prediction'].strip() for r in rows),
                'length_finished':sum(r['finish_reason']=='length' for r in rows),
                'actual_prompt_tokens_range':[min(r['usage']['prompt_tokens'] for r in rows),max(r['usage']['prompt_tokens'] for r in rows)],
                'output_tokens':total_tokens,'request_seconds':total_time,'tokens_per_request_second':total_tokens/total_time,
                'median_ttft':float(np.median([r['ttft'] for r in rows if r['ttft'] is not None])),
                'median_e2e':float(np.median([r['e2e'] for r in rows])),
                'median_post_first_rate':float(np.median(post)) if post else None,
                'max_content_gap':max(r['max_content_gap'] for r in rows)}
        comparisons[str(length)]={}
        for a,b in [('mixed','bf16_flash'),('mixed','bf16_triton'),('mixed','fp8'),('fp8','bf16_triton'),('bf16_triton','bf16_flash')]:
            diffs=[];regressed=[];improved=[];full_to_partial=[];partial_to_full=[]
            for task in tasks:
                task_diff=[]
                for sample in range(n):
                    key=f'{length}:{task}:{sample}';ra,rb=byarm[a][key],byarm[b][key]
                    delta=ra['score']-rb['score'];task_diff.append(delta)
                    if delta<0:regressed.append(key)
                    if delta>0:improved.append(key)
                    if rb['score']==1 and ra['score']<1:full_to_partial.append(key)
                    if ra['score']==1 and rb['score']<1:partial_to_full.append(key)
                    if delta:
                        audit.append({'length':length,'comparison':f'{a}-minus-{b}','id':key,'difference':delta,
                            'answers':data[key]['outputs'],a:ra['prediction'],b:rb['prediction']})
                diffs.append(task_diff)
            ci=paired_interval(diffs)
            name=f'{a}-minus-{b}'
            comparisons[str(length)][name]={'difference_pp':float(np.mean(diffs)*100),'ci95_pp':ci,
                'regressed_count':len(regressed),'improved_count':len(improved),'fully_correct_to_incomplete':len(full_to_partial),
                'incomplete_to_fully_correct':len(partial_to_full),'regressed_ids':regressed,'improved_ids':improved,
                'noninferiority_claim':False,
                'e2e_sum_ratio':results[str(length)][a]['request_seconds']/results[str(length)][b]['request_seconds']}
            all_deltas.setdefault(name,[]).append(np.asarray(diffs))
    aggregate={name:{'difference_pp':float(np.mean(values)*100),'ci95_pp':paired_interval(np.mean(values,axis=0))}
               for name,values in all_deltas.items()}
    ledger=read_jsonl(ROOT/'ledger.jsonl')
    output={'selection':selection,'results':results,'comparisons':comparisons,'aggregate':aggregate,
        'lifecycle_seconds':sum(r['seconds'] for r in ledger if r['event']=='end'),
        'formal_requests':4*len(tasks)*len(LENGTHS)*n,'data_sha256':digest(ROOT/'data-manifest.json')}
    write_json(ROOT/'analysis.json',output);write_json(ROOT/'answer-differences.json',audit)
    lines=['# RULER v1：mixed KV 质量与效率取舍','',
        f'固定 Qwen3-4B，13 个任务 × 3 个长度 × 每任务 {n} 题 × 4 个实现，共 {output["formal_requests"]} 个正式响应。非思考模式、单并发冷输入、等 4 GiB KV 字节预算。',
        '', '这是本地小样本评测，不是完整排行榜成绩。官方多答案任务可给部分分；分数不等于完整答对题目的比例。', '',
        '| 长度 | 实现 | RULER 分数 | 完全答对 | 输出截断 | TTFT 中位数(s) | E2E 中位数(s) | 输出token/请求总秒 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for length in LENGTHS:
        for arm in ARMS:
            r=results[str(length)][arm]
            lines.append(f'| {length} | {arm} | {r["score"]:.2f} | {r["fully_correct"]}/{13*n} | {r["length_finished"]} | {r["median_ttft"]:.3f} | {r["median_e2e"]:.3f} | {r["tokens_per_request_second"]:.2f} |')
    lines += ['', '## 配对分数差', '', '差值为前者减后者；单位为百分点。区间是按任务分层的 10000 次配对 bootstrap。', '',
              '| 长度 | 对比 | 分数差 | 95%区间 | 分数退化/改善题数 | 完全答对变不完整 |', '|---|---|---:|---|---|---:|']
    for length,items in comparisons.items():
        for name,r in items.items():
            lines.append(f'| {length} | {name} | {r["difference_pp"]:+.2f} | [{r["ci95_pp"][0]:+.2f}, {r["ci95_pp"][1]:+.2f}] | {r["regressed_count"]}/{r["improved_count"]} | {r["fully_correct_to_incomplete"]} |')
    lines += ['', '## 各任务分数', '', '| 长度 | 任务 | BF16 Flash | BF16 Triton | FP8 | Mixed |','|---|---|---:|---:|---:|---:|']
    for length in LENGTHS:
        for task in tasks:
            values=' | '.join(f'{results[str(length)][arm]["task_scores"][task]:.2f}' for arm in ARMS)
            lines.append(f'| {length} | {task} | {values} |')
    lines += ['', '## 测量边界', '',
        '- 输出采用正常 EOS 与官方任务上限；截断不自动当成服务错误，需结合逐题响应判断。模型生成长度不同，因此真实任务耗时不能当作等输出长度的 kernel 加速倍率。',
        '- 本轮不覆盖热/部分命中、多并发混合到达或所有模型能力。首内容后速率和内容块间隔包含 HTTP/SSE 效应。',
        '- 本轮估计能力差异，不以通过某个无损门槛为目标。小样本 bootstrap 不包含未观察到的失败类型；即使逐题相同、区间退化为零，也不能据此证明总体无损。',
        '- 宏平均掩盖的能力变化应结合各任务分数和逐题退化/改善检查。',
        f'- GPU 进程生命周期账本合计 {output["lifecycle_seconds"]/60:.2f} 分钟，上限 120 分钟。',
        '- 原始输入、SSE、配置、指标、路由日志、逐题差异和机器结果保存在 `.local/research/ruler-tradeoff-2026-09-16/`；协议见 [ruler-protocol.md](ruler-protocol.md)。']
    (WORKSPACE/'docs/ruler-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'aggregate':aggregate,'lifecycle_minutes':output['lifecycle_seconds']/60,'requests':output['formal_requests']},indent=2))

if __name__=='__main__':main()
