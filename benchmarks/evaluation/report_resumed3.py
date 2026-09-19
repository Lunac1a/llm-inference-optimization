"""Continuation report; preserves original failed attempt and timing caveats."""
import importlib.util
import statistics
import sys
import numpy as np
from config import *
from resume3 import RESUME, PREV, verify_original, verify_resume, verify_previous, verify_paused


def official_score():
    sys.path.insert(0, str(WORKSPACE / 'benchmarks/ruler'))
    import common
    common.UPSTREAM = UPSTREAM
    spec = importlib.util.spec_from_file_location('official_report_helpers', WORKSPACE / 'benchmarks/ruler/report.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.score


def paired_interval(delta):
    """Task x sample paired bootstrap; input values are fractional score differences."""
    delta = np.asarray(delta, dtype=float)
    assert delta.ndim == 2 and delta.shape[1] > 0
    rng = np.random.default_rng(42)
    indices = rng.integers(delta.shape[1], size=(10000, delta.shape[0], delta.shape[1]))
    draws = delta[np.arange(delta.shape[0])[None, :, None], indices].mean(axis=(1, 2)) * 100
    return np.quantile(draws, [.025, .975]).tolist()


def validate_records(records, expected, fixed=False):
    assert len(records) == len(expected) and {r['id'] for r in records} == set(expected)
    assert len({r['id'] for r in records}) == len(records)
    for row in records:
        source = expected[row['id']]
        assert row['done'] and row['prediction'].strip()
        assert row['finish_reason'] in ['stop', 'length']
        assert row['cached_tokens'] == row['cache_hit_delta'] == row['preemptions'] == 0
        assert row['usage']['prompt_tokens'] == len(source['prompt_ids'])
        assert 0 < row['usage']['completion_tokens'] <= source['max_tokens']
        if fixed: assert row['usage']['completion_tokens'] == source['max_tokens']


def main():
    verify_original(); verify_previous(); verify_paused(); verify_resume()
    completion = read(RESUME / 'completion.json')
    assert completion['complete'] and completion['port_released']
    assert completion['completed_quality_responses'] == 2340
    initial = rows(ROOT / 'ledger.jsonl'); continued = rows(RESUME / 'ledger.jsonl')
    assert len(initial) == 10 and initial[-1]['job'] == '8192-0-bf16_flash' and initial[-1]['reason'] == 'child_failed'
    paused = rows(PREV / 'ledger.jsonl')
    assert len(paused) == 8 and paused[-1]['reason'] == 'interrupted'
    assert len(continued) == 3 and all(r['reason'] == 'completed' and r['code'] == 0 for r in initial[:-1] + paused[:-1] + continued)
    failed_resume = rows(ROOT / 'resume-1/ledger.jsonl')
    assert len(failed_resume) == 1 and failed_resume[0]['reason'] == 'child_failed'
    ledger = initial + failed_resume + paused + continued
    assert sum(r['seconds'] for r in ledger) <= BUDGET_SECONDS
    numeric = read(ROOT / 'numerical.json'); assert len(numeric) == 20 and all(r['passed'] for r in numeric)
    data = {r['id']: r for p in (ROOT / 'data').glob('*/*.jsonl') for r in rows(p)}
    tasks = sorted({r['task'] for r in data.values()}); assert len(data) == 780 and len(tasks) == 13
    score = official_score(); results = {}; comparisons = {}; differences = []; scored = []; aggregate = {}
    deltas_by_comparison = {}; timing = {}
    for length in LENGTHS:
        results[str(length)] = {}; mapping = {}; timing[str(length)] = {}
        expected = {key: r for key, r in data.items() if r['target_length'] == length}
        for arm in ARMS:
            records = []; times = []; capacities = []; blocks = []
            for block in [0, 1]:
                job = f'{length}-{block}-{arm}'
                if job == '16384-1-fp8':
                    outs = [PREV / 'runs' / job, RESUME / 'runs' / job]
                else:
                    outs = [next(base / 'runs' / job for base in [RESUME, PREV, ROOT]
                                 if (base / 'runs' / job / 'summary.json').exists())]
                quality = []; latency = []; segment_capacities = []
                for out in outs:
                    q = rows(out / 'quality.jsonl')
                    lt = rows(out / 'latency.jsonl') if (out / 'latency.jsonl').exists() else []
                    warm = rows(out / 'warmup.jsonl'); assert len(warm) == 2
                    for record in q + lt + warm:
                        raw = rows(out / 'responses' / f"{record['raw_name']}.jsonl")
                        assert raw[-1]['data'] == '[DONE]'
                        assert (out / 'metrics' / f"{record['raw_name']}-before.txt").exists()
                        assert (out / 'metrics' / f"{record['raw_name']}-after.txt").exists()
                    quality.extend(q); latency.extend(lt)
                    import re
                    logs = '\n'.join(p.read_text(errors='replace') for p in (out / 'engine').glob('*.log'))
                    cap = re.findall(r'GPU KV cache size: ([\d,]+) tokens', logs)
                    assert cap
                    segment_capacities.append(int(cap[-1].replace(',', '')))
                assert len(quality) == 130 and len(latency) == 3 and len(set(segment_capacities)) == 1
                if len(outs) == 2:
                    assert len(rows(outs[0] / 'quality.jsonl')) == 24
                    assert len(rows(outs[1] / 'quality.jsonl')) == 106
                block_expected = {key: r for key, r in expected.items() if block * 10 <= r['sample'] < (block + 1) * 10}
                validate_records(quality, block_expected)
                timing_source = {**read(ROOT / 'timing-inputs.json')[str(length)], 'max_tokens': 32}
                for row in latency: validate_records([row], {row['id']: timing_source}, fixed=True)
                capacities.append(segment_capacities[0])
                records.extend(quality); times.extend(latency)
                blocks.append({'block': block, 'ttft_median': statistics.median(r['ttft'] for r in latency),
                    'e2e_median': statistics.median(r['e2e'] for r in latency),
                    'ttft_samples': [r['ttft'] for r in latency], 'e2e_samples': [r['e2e'] for r in latency]})
            validate_records(records, expected)
            assert len(set(capacities)) == 1
            mapping[arm] = {r['id']: r for r in records}
            for row in records:
                row['score'] = score(row['prediction'], data[row['id']]['outputs'], data[row['id']]['category'])
                scored.append({k: row[k] for k in ['id', 'task', 'sample', 'target_length', 'score', 'finish_reason', 'ttft', 'e2e']}
                              | {'arm': arm, 'output_tokens': row['usage']['completion_tokens']})
            task_scores = {task: float(np.mean([r['score'] for r in records if r['task'] == task]) * 100) for task in tasks}
            results[str(length)][arm] = {'score': float(np.mean(list(task_scores.values()))), 'task_scores': task_scores,
                'requests': len(records), 'fully_correct': sum(r['score'] == 1 for r in records),
                'truncated': sum(r['finish_reason'] == 'length' for r in records), 'empty': 0,
                'request_seconds': sum(r['e2e'] for r in records), 'output_tokens': sum(r['usage']['completion_tokens'] for r in records),
                'median_ttft': statistics.median(r['ttft'] for r in records), 'median_e2e': statistics.median(r['e2e'] for r in records),
                'capacity_tokens': capacities[0]}
            timing[str(length)][arm] = {'blocks': blocks, 'ttft_median': statistics.median(r['ttft'] for r in times),
                                       'e2e_median': statistics.median(r['e2e'] for r in times), 'requests': len(times)}
        comparisons[str(length)] = {}
        for base in ['bf16_flash', 'fp8']:
            delta = [[mapping['mixed'][f'{length}:{task}:{i}']['score'] - mapping[base][f'{length}:{task}:{i}']['score']
                      for i in range(N)] for task in tasks]
            difference = float(np.mean(delta) * 100)
            mixed_result = results[str(length)]['mixed']; base_result = results[str(length)][base]
            entry = {'difference_pp': difference, 'ci95_pp': paired_interval(delta),
                'relative_score_change_percent': 100 * difference / base_result['score'],
                'regressed': int(np.sum(np.asarray(delta) < 0)), 'improved': int(np.sum(np.asarray(delta) > 0)),
                'quality_request_time_change_percent': (mixed_result['request_seconds'] / base_result['request_seconds'] - 1) * 100,
                'fixed_output_ttft_change_percent': (timing[str(length)]['mixed']['ttft_median'] / timing[str(length)][base]['ttft_median'] - 1) * 100,
                'fixed_output_e2e_change_percent': (timing[str(length)]['mixed']['e2e_median'] / timing[str(length)][base]['e2e_median'] - 1) * 100}
            comparisons[str(length)][base] = entry
            deltas_by_comparison.setdefault(base, []).append(np.asarray(delta))
            for key in expected:
                a, b = mapping['mixed'][key], mapping[base][key]
                if a['score'] != b['score']:
                    differences.append({'id': key, 'baseline': base, 'mixed_score': a['score'], 'baseline_score': b['score'],
                        'answers': data[key]['outputs'], 'mixed_prediction': a['prediction'], 'baseline_prediction': b['prediction']})
    for base, delta in deltas_by_comparison.items():
        average = np.mean(delta, axis=0)
        aggregate[base] = {'difference_pp': float(average.mean() * 100), 'ci95_pp': paired_interval(average)}
    summary = {'version': '0.1.3', 'quality_responses': 2340, 'unique_quality_inputs': 780,
        'timing_responses': 54, 'results': results, 'comparisons': comparisons, 'aggregate': aggregate,
        'fixed_output_timing': timing, 'original_failed_attempt_preserved': True,
        'timing_caveats': {'4096': 'pre-resume Modern Standby exposure', '8192': 'spans interrupted and resumed segments', '16384': 'block 0 entirely resume-2; block 1 FP8 timing in resume-2 and BF16/mixed timing in resume-3; cross-segment descriptive comparison'},
        'gpu_lifecycle_seconds': sum(r['seconds'] for r in ledger),
        'data_manifest_sha256': digest(ROOT / 'data-manifest.json'), 'frozen_manifest_sha256': digest(ROOT / 'frozen.json')}
    write(ROOT / 'analysis.json', summary); write(ROOT / 'answer-differences.json', differences)
    export = WORKSPACE / 'benchmarks/results/final-2026-09-19'; export.mkdir(parents=True, exist_ok=True)
    write(export / 'summary.json', summary); write(export / 'scores.json', scored)
    write(export / 'answer-differences.json', differences)
    write(export / 'provenance.json', {'frozen_hashes': read(ROOT / 'frozen.json'), 'environment': read(ROOT / 'environment.json'),
        'completion': completion, 'numerical_cases': 20, 'full_hash_audit_passed': True,
        'resume_frozen_hashes': read(RESUME / 'frozen.json'), 'original_ledger': initial, 'failed_resume_ledger': failed_resume, 'paused_resume_ledger': paused, 'resume_ledger': continued, 'paused_frozen_hashes': read(PREV / 'frozen.json'),
        'power_guard': read(RESUME / 'power-guard.json')})
    lines = ['# 最终 mixed 0.1.3：扩大 RULER 评测', '',
        '13任务 × 3长度 × 20独立新样本 × 3实现，共2340正式响应；另有54次固定32-token延迟测量。单并发冷输入、等4 GiB KV预算。', '',
        '| 长度 | 实现 | RULER分数 | 完全答对/260 | 截断 | 请求总耗时(s) | TTFT中位数(s) | KV tokens |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for length in LENGTHS:
        for arm in ARMS:
            r = results[str(length)][arm]
            lines.append(f'| {length} | {arm} | {r["score"]:.3f} | {r["fully_correct"]} | {r["truncated"]} | {r["request_seconds"]:.2f} | {r["median_ttft"]:.3f} | {r["capacity_tokens"]} |')
    lines += ['', '## 配对质量差与耗时变化（时间结果仅作描述，含跨段比较）', '',
        '差值为 mixed 减对照。分数差为百分点，括号为相对百分比；区间按任务分层配对bootstrap 10000次。正常EOS耗时受输出长度影响。', '',
        '| 长度 | 对照 | 分数差 pp（相对%） | 95% CI pp | 退化/改善题数 | 正常EOS总耗时变化 | 固定32token TTFT变化 |',
        '|---|---|---:|---|---|---:|---:|']
    for length in LENGTHS:
        for base in ['bf16_flash', 'fp8']:
            r = comparisons[str(length)][base]
            lines.append(f'| {length} | {base} | {r["difference_pp"]:+.3f} ({r["relative_score_change_percent"]:+.3f}%) | {r["ci95_pp"]} | {r["regressed"]}/{r["improved"]} | {r["quality_request_time_change_percent"]:+.1f}% | {r["fixed_output_ttft_change_percent"]:+.1f}% |')
    lines += ['', '## 各任务分数', '', '| 长度 | 任务 | BF16 Flash | FP8 | Mixed |', '|---|---|---:|---:|---:|']
    for length in LENGTHS:
        for task in tasks:
            values = ' | '.join(f'{results[str(length)][a]["task_scores"][task]:.2f}' for a in ARMS)
            lines.append(f'| {length} | {task} | {values} |')
    lines += ['', '## 固定输出延迟（待机及跨段限制见下文）', '', '每块3次测量；两块反向组别顺序。以下为首个非空SSE内容时间，不是设备端精确token延迟。', '',
              '| 长度 | 实现 | 第一块中位数(s) | 第二块中位数(s) | 合并中位数(s) |', '|---|---|---:|---:|---:|']
    for length in LENGTHS:
        for arm in ARMS:
            t = timing[str(length)][arm]
            lines.append(f'| {length} | {arm} | {t["blocks"][0]["ttft_median"]:.3f} | {t["blocks"][1]["ttft_median"]:.3f} | {t["ttft_median"]:.3f} |')
    lines += ['', '## 范围与完整性', '',
        f'- GPU生命周期合计 {summary["gpu_lifecycle_seconds"] / 60:.2f} 分钟；三小时预算。20个数值用例、2340质量响应、54延迟响应和冻结文件哈希均已核验。',
        '- 这是本地20题/任务/长度评测，不是完整RULER排行榜；置信区间不包含未见错误类型，不证明普遍无损。',
        '- 用户授权续跑保留原1040响应、24延迟记录及BF16热身超时失败；resume-2保存7个完整块和FP8第二块24个响应；resume-3仅补齐其余366个响应。前两次失败及主动暂停记录均保留，未重跑已完成数据。',
        '- 4K包含Windows现代待机风险，8K跨越中断前后，时间只作描述，不能用于强加速结论。16K第一块在resume-2完成，第二块跨resume-2/3，不能把合并时间差解释为严格控制环境下的因果收益；另核查两段宿主电源事件。',
        '- 固定输出延迟只覆盖单输入/长度、单并发；交错顺序不消除设备状态漂移，不外推多并发、生产p95或所有模型。',
        '- 780个新输入不与旧输入重合；不同长度的QA问题可相同，整体区间保留任务/序号配对。',
        '- 缓存token容量不是总显存节省或已验证吞吐。没有运行新优化或依据结果调整参数。',
        '- 精简机器结果、逐题得分、答案差异与来源哈希见 [benchmarks/results/final-2026-09-19](../benchmarks/results/final-2026-09-19/)。',
        '- 完整输入/原始SSE/指标/日志/源码快照保存在 `.local/research/ruler-final-2026-09-19/`，不随Git自动发布。',
        '- 协议：[final-benchmark-protocol.md](final-benchmark-protocol.md)。']
    (WORKSPACE / 'docs/final-benchmark-results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'quality_responses': 2340, 'timing_responses': 54, 'aggregate': aggregate}, indent=2))


if __name__ == '__main__': main()
