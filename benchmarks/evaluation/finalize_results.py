"""CPU-only presentation/provenance supplement after the frozen report succeeds."""
import json
from pathlib import Path
import statistics

W = Path(__file__).resolve().parents[2]
ROOT = W / '.local/research/ruler-final-2026-09-19'
EXPORT = W / 'benchmarks/results/final-2026-09-19'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p, x): p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def main():
    summary = read(EXPORT/'summary.json')
    completion = read(ROOT/'resume-3/completion.json')
    guard = read(ROOT/'resume-3/power-guard.json')
    assert completion['complete'] and completion['port_released'] and not guard['active']
    assert len(completion['compute_processes_after'].strip().splitlines()) == 1
    assert summary['quality_responses'] == 2340 and summary['timing_responses'] == 54
    scores = {arm: statistics.mean(summary['results'][str(length)][arm]['score'] for length in [4096,8192,16384])
              for arm in ['bf16_flash','fp8','mixed']}
    capacity_ratio = summary['results']['16384']['mixed']['capacity_tokens']/summary['results']['16384']['bf16_flash']['capacity_tokens']
    relative = (scores['mixed']/scores['bf16_flash']-1)*100
    summary['overall_scores'] = scores
    summary['overall_mixed_vs_bf16_relative_percent'] = relative
    summary['mixed_vs_bf16_capacity_ratio'] = capacity_ratio
    events = {name: read(ROOT/name/'host-power-events.json') for name in ['resume-2','resume-3']}
    summary['host_events_checked'] = {name: len(value) for name,value in events.items()}
    write(EXPORT/'summary.json', summary)
    write(ROOT/'analysis.json', summary)
    provenance = read(EXPORT/'provenance.json')
    provenance['host_event_audit'] = {'events': events, 'system_event_ids': [1,42,107,4101,506,507],
        'provider_filter': 'Power|Display|nvlddmkm', 'scope': 'Each continuation power-guard start through end; absence does not prove identical hardware conditions.'}
    write(EXPORT/'provenance.json', provenance)
    status = {'complete': True, 'state':'completed', 'quality_responses':2340, 'planned_quality_responses':2340,
        'unique_quality_inputs':780, 'fixed_output_measurements':54, 'planned_fixed_output_measurements':54,
        'numerical_cases_passed':20, 'gpu_lifecycle_seconds':completion['gpu_lifecycle_seconds'],
        'budget_seconds':10800, 'remaining_budget_seconds':10800-completion['gpu_lifecycle_seconds'],
        'port_released':True, 'compute_processes_after':completion['compute_processes_after'],
        'power_guard_active':False, 'full_hash_audit_passed':True, 'original_failures_and_pause_preserved':True,
        'repeated_completed_quality_requests':0, 'timing_caveats':summary['timing_caveats']}
    write(EXPORT/'status.json', status)
    report = W/'docs/final-benchmark-results.md'
    content = report.read_text(encoding='utf-8')
    marker = '<!-- final-interpretation -->'
    if marker not in content:
        paragraphs = f'''{marker}
评测已完成，累计GPU时间149.75分钟，包含此前失败与续跑，未超过三小时预算。服务和GPU已释放。

| 总体指标 | BF16 Flash | 原生 FP8 | Mixed |
|---|---:|---:|---:|
| 三长度等权平均RULER分数（满分100） | {scores['bf16_flash']:.3f} | {scores['fp8']:.3f} | {scores['mixed']:.3f} |
| 4 GiB KV预算下token容量 | 29,120 | 56,480 | 56,480 |

Mixed总体分数比BF16相对下降 **{-relative:.3f}%**（绝对差0.235个百分点），缓存容量为 **{capacity_ratio:.2f}倍**。总体配对分数差95%区间为[-0.654, +0.094]个百分点，不能据此证明等效或完全无损。

16K上mixed分数91.487，对比BF16的92.436相对下降1.026%；差异主要来自QA_2和FWE，而非检索题。该长度配对差95%区间为[-2.141, -0.013]个百分点，不能用总体均值掩盖长上下文退化。

16K观察到mixed正常EOS请求总耗时比原生FP8少39.4%，但比BF16多19.6%；固定32-token输出的首内容延迟分别少45.5%、多36.3%。这些是本次测量结果，不是对全部工作负载的加速保证：第二数据块跨运行时段，正常EOS耗时也受输出长度影响。resume-2与resume-3的宿主日志均未检出所检查的待机/恢复/显卡事件，但这不能消除设备状态差异。

因此，本项目证据支持“接近BF16的总体质量、FP8级别的KV容量、比原生FP8更低的本地长输入耗时”，不支持“比BF16更快且完全无损”。以下保留完整分项结果。
<!-- /final-interpretation -->

'''
        first, rest = content.split('\n\n', 1)
        content = first+'\n\n'+paragraphs+rest
        # Keep the raw precision in JSON; display intervals at a readable precision.
        import re
        content = re.sub(r'\[(-?\d+\.\d+), (-?\d+\.\d+)\]',
                         lambda m: f'[{float(m[1]):.3f}, {float(m[2]):.3f}]', content)
        report.write_text(content, encoding='utf-8')
    print(json.dumps({'overall_scores':scores,'relative_quality_change_percent':relative,
                      'capacity_ratio':capacity_ratio,'status':status},indent=2))

if __name__ == '__main__': main()
