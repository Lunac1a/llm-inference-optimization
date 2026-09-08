"""Combine completed measurement, integrity, budget and cleanup evidence."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/stage2'


def load(name): return json.loads((OUT/name).read_text())


def main():
    verify=load('verification.json');selection=load('selection.json')
    expected={f'sweep-{w}-c{c}' for w in 'ABC' for c in [1,2,4,8,16]}|{
        f'confirm-r{r}-c{c}' for r in [1,2,3] for c in selection['points']}|{
        f'contrast-r{r}-cap{c}' for r in [1,2,3] for c in [4,8]}
    actual={p.parent.name for p in OUT.glob('*/summary.json') if p.parent.name.startswith(('sweep-','confirm-','contrast-r'))}
    formal=[load(n+'/summary.json') for n in expected]
    integrity=(OUT/'stage1-integrity-final.log').read_text().splitlines()
    regression=(OUT/'regression-final.log').read_text()
    checks={'measurement_gates':verify['measurement_gates_passed'],'exact_formal_matrix':actual==expected,
            'formal_measured_requests_720':sum(r['completed'] for r in formal)==720,
            'original_stage1_integrity':bool(integrity) and all(s.endswith(': OK') for s in integrity),
            'installed_source_unchanged':all(load('installed-source-integrity.json').values()),
            'within_budget':load('budget-final.json')['within_90_minutes'],
            'services_clean':load('cleanup-verification.json')['cleanup_passed'],
            'profiles_cleanly_stopped':all((OUT/n/'stop-status.txt').read_text()=='200' for n in ['profile-low','profile-knee']),
            'regression_seven_passed':'Ran 7 tests' in regression and regression.rstrip().endswith('OK')}
    report={'stage2_passed':all(checks.values()),'checks':checks,
            'outcome':'Supported attribution to imposed admission cap for A/c8; no custom optimization claim.',
            'stage3_started':False,'cloud_spend_aud':0,'original_stage1_files_checked':len(integrity)}
    (OUT/'acceptance.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return 0 if report['stage2_passed'] else 1


if __name__=='__main__':raise SystemExit(main())
