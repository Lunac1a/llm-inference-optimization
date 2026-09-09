"""Read-only analysis of collected trials, plus a separate derived report."""
import ast
import hashlib
import json
from pathlib import Path
import re
import statistics
from stage3_lib import ROOT, write_json

OUT = ROOT / 'artifacts/mixed-load'


def analyze():
    table = []
    routes = {}
    quality = []
    for mode in ('full', 'chunk'):
        folder = OUT / mode
        if not folder.exists():
            continue
        log = (folder / 'server/server.log').read_text(errors='replace')
        schedules = []
        for line in log.splitlines():
            if 'HYBRID_SCHEDULE ' in line:
                schedules.append(ast.literal_eval(line.split('HYBRID_SCHEDULE ', 1)[1]))
        cold = [end - start for batch in schedules for start, end, length, direct in batch if direct]
        cached = [(end - start, length) for batch in schedules for start, end, length, direct in batch if not direct and end - start > 1]
        routes[mode] = {'schedule_batches': len(schedules), 'direct_bf16_query_lengths': sorted(set(cold)),
                        'cached_fp8_query_sequence_lengths': sorted(set(cached)),
                        'cache_tokens': re.findall(r'GPU KV cache size: ([\d,]+) tokens', log),
                        'mixed_decode_prefill_batches': sum(any(e - s == 1 for s, e, l, d in b) and
                                                           any(e - s > 1 for s, e, l, d in b) for b in schedules)}
        for path in sorted(folder.glob('r*/summary.json')):
            r = json.loads(path.read_text())
            telemetry = [json.loads(l) for l in (path.parent / 'telemetry.jsonl').read_text().splitlines()]
            def delta(name):
                return sum(x['delta'] for x in r['metric_deltas'] if x['name'] == name)
            def peak(name):
                values = [x['value'] for s in telemetry for x in s.get('metrics', {}).get('selected', []) if x['name'] == name]
                return max(values) if values else None
            query = delta('vllm:prefix_cache_queries_total')
            hit = delta('vllm:prefix_cache_hits_total')
            table.append({'mode': mode, 'trial': path.parent.name, 'A_gap': r['A_gap'], 'throughput': r['throughput'],
                'requests': r['requests'], 'queue_sum': delta('vllm:request_queue_time_seconds_sum'),
                'queue_count': delta('vllm:request_queue_time_seconds_count'), 'preemptions': r['preemptions'],
                'prefix_queries': query, 'prefix_hits': hit, 'prefix_hit_query': hit / query if query else None,
                'running_peak': peak('vllm:num_requests_running'), 'waiting_peak': peak('vllm:num_requests_waiting'),
                'active_kv_peak': peak('vllm:kv_cache_usage_perc')})
        for path in sorted(folder.glob('quality-*.json')):
            r = json.loads(path.read_text())
            quality.append({'mode': mode, 'document': path.stem, 'text': r['text'], 'finish': r['finish_reason']})
    analysis = {'trials': table, 'routes': routes, 'quality': quality}
    results_file = OUT / 'results.json'
    if results_file.exists():
        results = json.loads(results_file.read_text())
        analysis['decisions'] = {k: v['decision'] for k, v in results.items()}
    write_json(OUT / 'analysis.json', analysis)
    lines = ['# Mixed-load measurement table', '', 'Seconds; client SSE content gaps, not exact token intervals.', '',
             '| Mode / trial | A max gap | A E2E | B TTFT / E2E | C TTFT / E2E | Output tok/s | Queue sum | Active KV peak |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in table:
        a, b = r['requests']['A'], r['requests']['B']
        c = r['requests'].get('C')
        ctext = f"{c['ttft']:.3f} / {c['e2e']:.3f}" if c else 'absent'
        lines.append(f"| {r['mode']} / {r['trial']} | {r['A_gap']:.3f} | {a['e2e']:.3f} | {b['ttft']:.3f} / {b['e2e']:.3f} | {ctext} | {r['throughput']:.2f} | {r['queue_sum']:.6f} | {r['active_kv_peak']:.3f} |")
    (OUT / 'measurement-table.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == '__main__': analyze()
