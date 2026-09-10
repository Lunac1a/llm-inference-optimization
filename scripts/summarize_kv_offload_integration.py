"""Integration telemetry summary; skip servers stopped before performance."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from stage3_lib import ROOT, write_json, percentile
from finalize_mixed_load import verify
from local_document_qa import configuration, port_open
from validate_mixed_load import content_times

OUT = ROOT / 'artifacts/kv-offload-integration'
COMPAT = OUT


def read(path):
    return json.loads(path.read_text())


def main():
    runs, quality = [], []
    docs = read(OUT / 'materials.json')['documents']
    for folder in sorted(OUT.glob('r*-*')):
        if not (folder / 'measurement.json').exists():
            continue
        source = read(folder / 'measurement.json')
        requests = []
        for row in source['requests']:
            raw = read(folder / (row['phase'] + '-' + row['document'] + '.json'))
            times = content_times(raw)
            def total(name, direction=None):
                return sum(m['delta'] for m in row['metric_deltas'] if m['name'] == 'vllm:' + name
                           and (direction is None or m['labels'].get('transfer_type') == direction))
            transfer = {}
            for direction in ('GPU_to_CPU', 'CPU_to_GPU'):
                size = total('kv_offload_total_bytes_total', direction)
                seconds = total('kv_offload_total_time_total', direction)
                transfer[direction] = {'bytes': size, 'cuda_event_seconds': seconds,
                    'count': total('kv_offload_size_count', direction),
                    'effective_bytes_per_event_second': size / seconds if seconds else None}
            requests.append({'phase': row['phase'], 'document': row['document'], **row['request'],
                'controls': row['controls'], 'queue_seconds': total('request_queue_time_seconds_sum'),
                'computed_prefill_tokens': total('request_prefill_kv_computed_tokens_sum'),
                'local_queries': total('prefix_cache_queries_total'),
                'external_queries': total('external_prefix_cache_queries_total'),
                'max_sse_content_gap_seconds': max((b-a for a, b in zip(times, times[1:])), default=None),
                'transfer': transfer, 'finish_reason': raw['finish_reason'], 'done': raw['done']})
        telemetry = [json.loads(line) for line in (folder / 'telemetry.jsonl').read_text().splitlines()]
        occupancy = [m['value'] for t in telemetry for m in t.get('metrics', {}).get('selected', [])
                     if m['name'] == 'vllm:kv_cache_usage_perc']
        phases = {}
        for phase in ('cold', 'revisit'):
            rows = [r for r in requests if r['phase'] == phase]
            phases[phase] = {'count': len(rows), 'complete_wave': len(rows) == 3,
                'ttft_median': percentile([r['ttft'] for r in rows], 50),
                'ttft_p95': percentile([r['ttft'] for r in rows], 95),
                'e2e_p95': percentile([r['e2e'] for r in rows], 95),
                'output_tokens_per_request_seconds': sum(r['token_count'] for r in rows) / sum(r['e2e'] for r in rows),
                'output_tokens_per_observed_wave_second': sum(r['token_count'] for r in rows) / (rows[-1]['end']-rows[0]['start'])}
        runs.append({'run': folder.name, 'capacity': read(folder / 'capacity.json'), 'requests': requests,
            'phases': phases, 'sampled_max_owned_rss_bytes': max(t['rss']['sum_rss_bytes'] for t in telemetry),
            'sampled_min_mem_available_bytes': min(t['memory']['MemAvailable'] for t in telemetry),
            'sampled_max_swap_used_bytes': max(t['memory']['swap_used'] for t in telemetry),
            'sampled_max_active_gpu_cache_fraction': max(occupancy) if occupancy else None})
        for path in sorted(folder.glob('quality-*.json')):
            r = read(path)
            key = path.stem.removeprefix('quality-')
            facts = docs[key]['facts']
            # Exact-value helper for these observed answers; human body review is in validation.
            correct = all(facts[k] in r['text'] and facts[k] in docs[key]['text'] for k in ('owner', 'budget'))
            foreign = any(d['facts'][k] in r['text'] for other, d in docs.items() if other != key for k in ('owner', 'budget'))
            quality.append({'run': folder.name, 'document': key, 'text': r['text'], 'expected': facts,
                'normal_eos': r['finish_reason'] == 'stop', 'exact_values_present': correct, 'foreign_values': foreign})
    write_json(OUT / 'analysis.json', {'result': 'stopped_on_baseline_cache_control', 'accepted': False,
        'completed_pairs': 0, 'performance_requests': sum(len(r['requests']) for r in runs),
        'quality_requests': len(quality), 'warmup_requests': len(list(OUT.glob('r*-*/warm-*.json'))),
        'runs': runs, 'quality': quality,
        'timing_scope': 'SSE content arrivals, not exact token generation; partial waves are not acceptance samples'})
    tests = []
    for pattern in ('test_kv_offload.py', 'test_kv_offload_transport.py', 'test_mixed_load.py', 'test_hybrid_prefill.py'):
        r = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests'), '-p', pattern],
                           capture_output=True, text=True, timeout=60)
        (OUT / (pattern + '.log')).write_text(r.stdout + r.stderr)
        tests.append({'pattern': pattern, 'returncode': r.returncode})
    closure = {'old_evidence': verify(read(OUT / 'old-evidence-hashes.json')),
        'pre_diagnosis_evidence': verify(read(COMPAT / 'old-evidence-hashes.json')),
        'frozen_sources': verify(read(OUT / 'frozen-source-hashes.json')),
        'installed_sources': verify(read(OUT / 'preflight.json')['package_hashes'], True),
        'tests': tests, 'port_closed': not port_open(configuration()),
        'compute_processes': subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv'],
                                          capture_output=True, text=True).stdout,
        'gpu_retry': False, 'accepted': False}
    write_json(OUT / 'closure.json', closure)
    assert all(closure[k]['pass'] for k in ('old_evidence', 'pre_diagnosis_evidence', 'frozen_sources', 'installed_sources'))
    assert all(t['returncode'] == 0 for t in tests)
    assert closure['port_closed'] and closure['compute_processes'].strip() == 'pid, process_name'
    for folder in (COMPAT, OUT):
        write_json(folder / 'sha256-manifest.json', {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob('*') if p.is_file() and p.name != 'sha256-manifest.json'})
    print(json.dumps(closure, indent=2))


if __name__ == '__main__':
    main()
