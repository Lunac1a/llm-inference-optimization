"""Read raw Kineto events without double-counting GPU annotations as kernels."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2] / '.local/research/mixed-latency-2026-09-18'


def trace_summary(path):
    events = json.loads(gzip.decompress(path.read_bytes()))['traceEvents']
    cpu = {e['args']['External id']: e for e in events if e.get('cat') == 'cpu_op'
           and 'External id' in e.get('args', {})}
    kernels = [e for e in events if e.get('cat') == 'kernel']
    assert kernels, 'Trace has no CUDA kernel events'
    categories = defaultdict(lambda: {'ms': 0, 'calls': 0})
    names = defaultdict(lambda: {'ms': 0, 'calls': 0})
    matrix_shapes = defaultdict(lambda: {'ms': 0, 'calls': 0})
    for e in kernels:
        name = e['name']; op = cpu.get(e.get('args', {}).get('External id'), {})
        if 'attention' in name.lower() or 'flash_fwd' in name.lower() or '_fwd_kernel' in name:
            group = 'attention'
        elif 'cache' in name.lower() and ('reshape' in name.lower() or 'quant' in name.lower()):
            group = 'kv_write'
        elif op.get('name') == 'aten::mm' or any(s in name.lower() for s in ['gemm', 'gemv', 'cutlass']):
            group = 'matrix_multiply'
            dims = op.get('args', {}).get('Input Dims', [])
            shape = str(dims[1] if len(dims) > 1 else 'unmapped')
            matrix_shapes[shape]['ms'] += e['dur'] / 1000
            matrix_shapes[shape]['calls'] += 1
        elif op.get('name') == 'aten::sub': group = 'metadata_subtraction'
        else: group = 'other'
        e['_group'] = group
        for bucket, key in [(categories, group), (names, name)]:
            bucket[key]['ms'] += e['dur'] / 1000; bucket[key]['calls'] += 1
    gpu = [e for e in events if e.get('cat') in ['kernel', 'gpu_memcpy', 'gpu_memset']]
    intervals = sorted((e['ts'], e['ts'] + e['dur']) for e in gpu)
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]: merged[-1][1] = max(end, merged[-1][1])
        else: merged.append([start, end])
    envelope = merged[-1][1] - merged[0][0]
    busy = sum(b - a for a, b in merged)
    gaps = sorted([(b[0] - a[1]) / 1000 for a, b in zip(merged, merged[1:])], reverse=True)
    operations = defaultdict(lambda: {'inclusive_ms': 0, 'calls': 0})
    for e in cpu.values():
        if e['name'] in ['aten::sub', 'aten::slice', 'aten::mm', 'vllm::unified_attention_with_output', 'vllm::unified_kv_cache_update']:
            operations[e['name']]['inclusive_ms'] += e['dur'] / 1000
            operations[e['name']]['calls'] += 1
    chunks = []
    for e in sorted((e for e in events if e.get('cat') == 'gpu_user_annotation'), key=lambda e: e['ts']):
        selected = [k for k in kernels if e['ts'] <= k['ts'] < e['ts'] + e['dur']]
        chunks.append({'name': e['name'], 'span_ms': e['dur'] / 1000,
                       'attention_ms': sum(k['dur'] for k in selected if k['_group'] == 'attention') / 1000})
    return {'trace': str(path.relative_to(ROOT)), 'categories': dict(categories),
        'kernel_sum_ms': sum(e['dur'] for e in kernels) / 1000,
        'gpu_envelope_ms': envelope / 1000, 'gpu_busy_union_ms': busy / 1000,
        'gpu_gap_ms': (envelope - busy) / 1000, 'largest_gaps_ms': gaps[:10],
        'matrix_weight_shapes': dict(matrix_shapes), 'cpu_inclusive_ops': dict(operations),
        'chunks': chunks,
        'top_kernels': sorted([{'name': k, **v} for k, v in names.items()], key=lambda x: -x['ms'])[:20]}


def main():
    summary = {}
    for arm in ['bf16_triton', 'mixed', 'fp8', 'bf16_flash']:
        out = ROOT / arm
        if not (out / 'summary.json').exists(): continue
        rows = [json.loads(line) for line in (out / 'results.jsonl').read_text().splitlines()]
        measures = {}
        for length in [4096, 8192, 16384]:
            data = [r for r in rows if r['phase'] == 'measure' and r['length'] == length]
            assert len(data) == 3
            samples = [r['ttft'] for r in data]
            measures[length] = {'samples': samples, 'median': statistics.median(samples),
                                'min': min(samples), 'max': max(samples), 'prompt_tokens': data[0]['prompt_tokens']}
        traces = list((out / 'traces').glob('*.pt.trace.json.gz')); assert len(traces) == 1
        summary[arm] = {'measurements': measures, 'profile': trace_summary(traces[0]),
                        **json.loads((out / 'summary.json').read_text())}
    (ROOT / 'analysis.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    for arm, result in summary.items():
        print(arm, json.dumps({'timing': result['measurements'], 'categories': result['profile']['categories'],
            'matrix_weight_shapes': result['profile']['matrix_weight_shapes'],
            'gpu_gap_ms': result['profile']['gpu_gap_ms']}))


if __name__ == '__main__': main()
