"""CPU-only acceptance of the fixed API integration experiment."""
import hashlib
import json
import subprocess
import sys
import summarize_kv_offload_integration as base
from stage3_lib import ROOT, write_json

OUT = ROOT / 'artifacts/kv-offload-integration'


def main():
    # Reuse raw-stream/telemetry summarization without touching historical paths.
    base.OUT = base.COMPAT = OUT
    base.main()
    analysis = base.read(OUT / 'analysis.json')
    pairs = []
    for n in range(1, 4):
        by = {r['run']: r for r in analysis['runs']}
        if any(f'r{n}-{mode}' not in by for mode in ('baseline', 'offload')):
            continue
        baseline, offload = [by[f'r{n}-{mode}'] for mode in ('baseline', 'offload')]
        waves = [r['phases'][p] for r in (baseline, offload) for p in ('cold', 'revisit')]
        if not all(w['complete_wave'] for w in waves):
            continue
        b, o = baseline['phases'], offload['phases']
        ttft_reduction = 1 - o['revisit']['ttft_p95'] / b['revisit']['ttft_p95']
        cold_regression = o['cold']['e2e_p95'] / b['cold']['e2e_p95'] - 1
        controls = base.read(OUT / f'r{n}-pair-control.json')
        gates = {'ttft_below_2s': o['revisit']['ttft_p95'] <= 2,
            'ttft_reduction_at_least_30pct': ttft_reduction >= .30,
            'revisit_e2e_no_regression': o['revisit']['e2e_p95'] <= b['revisit']['e2e_p95'],
            'cold_e2e_regression_at_most_15pct': cold_regression <= .15,
            'matched_retained_prefix': len(controls) == 3 and all(r['pass'] for r in controls),
            'request_controls': all(r['controls']['pass'] and r['done'] and r['token_count'] == 128
                                    for run in (baseline, offload) for r in run['requests'])}
        pairs.append({'round': n, 'ttft_reduction': ttft_reduction, 'cold_e2e_regression': cold_regression,
                      'gates': gates, 'pass': all(gates.values())})
    api = []
    for run in ('r1-offload', 'r3-offload'):
        p = OUT / run / 'api-nonstream.json'
        answer = base.read(p) if p.exists() else None
        api.append({'run': run, 'pass': bool(answer and answer['choices'][0]['finish_reason'] == 'stop'
                    and '服务正常' in answer['choices'][0]['message']['content'])})
    quality = len(analysis['quality']) == 6 and all(q['normal_eos'] and q['exact_values_present']
               and not q['foreign_values'] for q in analysis['quality'])
    accepted = (len(pairs) == 3 and all(p['pass'] for p in pairs) and quality
                and all(r['pass'] for r in api) and not (OUT / 'failure.json').exists())
    analysis.update({'result': 'accepted_bounded_serial_cpu_profile' if accepted else 'not_accepted',
        'accepted': accepted, 'completed_pairs': len(pairs), 'pairs': pairs, 'api_checks': api,
        'quality_value_checks': quality, 'quality_review': 'See validation document for human body review',
        'timing_scope': 'SSE content arrivals, not exact token generation; n=3 per wave, not population SLA'})
    write_json(OUT / 'analysis.json', analysis)
    closure = base.read(OUT / 'closure.json')
    result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests'),
                             '-p', 'test_kv_offload_integration.py'], capture_output=True, text=True, timeout=60)
    (OUT / 'test_kv_offload_integration.py.log').write_text(result.stdout + result.stderr)
    closure['tests'].append({'pattern': 'test_kv_offload_integration.py', 'returncode': result.returncode})
    closure['accepted'] = accepted and result.returncode == 0
    write_json(OUT / 'closure.json', closure)
    write_json(OUT / 'sha256-manifest.json', {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in OUT.rglob('*') if p.is_file() and p.name != 'sha256-manifest.json'})
    assert result.returncode == 0
    print(json.dumps({'accepted': closure['accepted'], 'pairs': pairs, 'api': api, 'quality': quality}, indent=2))


if __name__ == '__main__': main()
