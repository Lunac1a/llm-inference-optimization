"""Derived fixed-gate assessment and timing/progress audit. No requests."""
import json
import statistics
from stage3_lib import ROOT, write_json, percentile
from validate_mixed_load import content_times

OUT = ROOT / 'artifacts/mixed-load-rerun'


def report():
    analysis = json.loads((OUT / 'analysis.json').read_text())
    groups = {}
    for mode in ('full', 'chunk'):
        for condition in ('control', 'mixed'):
            rows = [r for r in analysis['trials'] if r['mode'] == mode and r['trial'].endswith(condition)]
            if not rows:
                continue
            fields = {'A_gap': [r['A_gap'] for r in rows], 'throughput': [r['throughput'] for r in rows],
                      'queue_sum': [r['queue_sum'] for r in rows]}
            gaps = [json.loads((OUT / mode / r['trial'] / 'summary.json').read_text())['A_gaps'] for r in rows]
            fields['A_gap_p95'] = [percentile(g, 95) for g in gaps]
            fields['A_gaps_ge_250ms'] = [sum(x >= .250 for x in g) for g in gaps]
            fields['A_time_in_gaps_ge_250ms'] = [sum(x for x in g if x >= .250) for g in gaps]
            for req in ('A', 'B', 'C'):
                for metric in ('ttft', 'e2e', 'output_tok_s'):
                    values = [r['requests'][req][metric] for r in rows if req in r['requests']]
                    if values:
                        fields[req + '_' + metric] = values
            groups[mode + '-' + condition] = {k: {'values': v, 'median': statistics.median(v)} for k, v in fields.items()}
    proof = []
    lengths = []
    for folder in sorted(OUT.glob('*/r*')):
        if not (folder / 'summary.json').exists():
            continue
        s = json.loads((folder / 'summary.json').read_text())
        a = json.loads((folder / 'A.json').read_text())
        dispatch = s['dispatch']
        count_before = sum(len(c.get('token_ids') or []) for e in a['events'] if e['time'] <= dispatch
                           for c in e['data'].get('choices', []))
        finish = [e['time'] for e in a['events'] if any(c.get('finish_reason') for c in e['data'].get('choices', []))]
        proof.append({'trial': str(folder.relative_to(OUT)), 'trigger_tokens': s['trigger']['A_tokens'],
                      'A_received_tokens_at_dispatch': count_before,
                      'trigger_to_dispatch_ms': (dispatch - s['trigger']['time']) * 1000,
                      'A_finish_after_dispatch': min(finish) > dispatch,
                      'A_remaining_received_tokens': a['token_count'] - count_before})
        for req in ('A', 'B', 'C'):
            path = folder / (req + '.json')
            if path.exists():
                r = json.loads(path.read_text())
                lengths.append({'trial': str(folder.relative_to(OUT)), 'request': req, 'output_tokens': r['token_count'],
                                'content_events': len(content_times(r)), 'done': r['done'], 'finish': r['finish_reason'],
                                'cached_tokens': r['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0)})
    result = {'groups': groups, 'dispatch_proof': proof, 'performance_streams': lengths,
              'decisions': analysis.get('decisions'), 'quality_review_artifact': 'answer-review.md',
              'quality_note': 'Separate assistant source inspection; not automatically scored here'}
    if 'chunk-mixed' in groups:
        full, chunk = groups['full-mixed'], groups['chunk-mixed']
        changes = {k: (chunk[k]['median'] / full[k]['median'] - 1) * 100 for k in chunk if k in full and full[k]['median']}
        flags = analysis['decisions']['full']['pairs']
        affected = [key for key in ('A', 'B') if any(p[key] for p in flags)]
        reductions = {key: changes['A_gap' if key == 'A' else 'B_ttft'] <= -30 for key in affected}
        result['mitigation'] = {'affected_metrics': affected, 'median_changes_percent': changes,
            'latency_reductions_pass': all(reductions.values()), 'reductions': reductions,
            'A_B_e2e_gate_pass': changes['A_e2e'] <= 15 and changes['B_e2e'] <= 15,
            'zero_preemptions': all(r['preemptions'] == 0 for r in analysis['trials'])}
    write_json(OUT / 'assessment.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('dispatch_proof', 'performance_streams')}, indent=2))


if __name__ == '__main__': report()
