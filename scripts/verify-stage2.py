"""Fail-closed Stage 2 evidence accounting; never certify partial execution."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/stage2'


def read(name):
    p=OUT/name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    expected={f'sweep-{w}-c{c}' for w in 'ABC' for c in [1,2,4,8,16]}
    actual={p.parent.name for p in OUT.glob('sweep-*/summary.json')}
    rows=[json.loads(p.read_text()) for p in OUT.glob('*/summary.json')]
    all_requests=True;raw_count=0
    for p in OUT.glob('*/requests.jsonl'):
        requests=[json.loads(s) for s in p.read_text().splitlines()]
        raw_count+=len(requests)
        all_requests &= all(r['success'] and not r['error'] for r in requests)
    confirm=read('confirm-stability.json');repeat=read('confirm-repeat-stability.json')
    chosen='confirm' if confirm and confirm['passed'] else 'confirm-repeat' if repeat and repeat['passed'] else None
    traces=read('trace-analysis.json') or []
    cuda=all(any(label in r['file'] and r['cuda_events_verified'] for r in traces) for label in ['profile-low','profile-knee'])
    frozen={p.replace('\\','/'):v for p,v in read('formal-source-sha256.json').items()}
    collection=['scripts/stage2-run.py','scripts/stage2-bench-entry.py','scripts/stage2-experiment.py',
                'scripts/benchmark-point.sh','scripts/run-stage1-benchmark.sh','scripts/analyze-stage1-benchmark.py',
                'configs/stage2-cap4.env','configs/stage2-cap8.env']
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==frozen[p] for p in collection}
    contrast=read('contrast-summary.json')
    report={'sweep_complete':actual==expected,'sweep_points':len(actual),'selected_confirmation':chosen,
            'all_raw_requests_successful':all_requests,'all_summarized_lengths_valid':all(r['request_evidence_passed'] for r in rows),
            'total_recorded_requests_including_preparation_warmups_profiles':raw_count,
            'collector_hashes_unchanged':hashes,'both_profiles_have_cuda_events':cuda,
            'contrast_complete':bool(contrast and len(contrast['pairs'])==3),'contrast_stable':bool(contrast and contrast['stable'])}
    report['measurement_gates_passed']=bool(report['sweep_complete'] and chosen and all_requests and
        report['all_summarized_lengths_valid'] and all(hashes.values()) and cuda and report['contrast_complete'] and report['contrast_stable'])
    report['note']='Scientific conclusion, Stage 1 integrity, budget and cleanup must also be reviewed in the validation document; this report alone does not certify the stage.'
    (OUT/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return 0 if report['measurement_gates_passed'] else 1


if __name__=='__main__':raise SystemExit(main())
