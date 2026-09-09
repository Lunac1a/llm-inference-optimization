"""CPU-only closure, source audit and preservation checks; no server launch."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from stage3_lib import ROOT, write_json
from local_document_qa import port_open
from validate_hybrid_prefill import config

OUT = ROOT / 'artifacts/mixed-load'


def verify(mapping, absolute=False):
    bad = []
    for name, expected in mapping.items():
        path = Path(name) if absolute else ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            bad.append(name)
    return {'count': len(mapping), 'changed_or_missing': bad, 'pass': not bad}


def main():
    pre = json.loads((OUT / 'preflight.json').read_text())
    package = next(Path(p).parents[1] for p in pre['package_hashes'] if p.endswith('/config/scheduler.py'))
    sources = ['entrypoints/openai/api_server.py', 'entrypoints/serve/dev/cache/api_router.py', 'envs.py']
    write_json(OUT / 'reset-api-source-audit.json', {
        'sources': {s: (package / s).read_text() for s in sources},
        'sha256': {s: hashlib.sha256((package / s).read_bytes()).hexdigest() for s in sources}})
    tests = []
    for pattern in ('test_mixed_load.py', 'test_hybrid_prefill.py'):
        result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests'), '-p', pattern],
                                capture_output=True, text=True, timeout=60)
        (OUT / (pattern + '.log')).write_text(result.stdout + result.stderr)
        tests.append({'pattern': pattern, 'returncode': result.returncode})
    closure = {'old_evidence': verify(json.loads((OUT / 'old-evidence-hashes.json').read_text())),
               'frozen_sources': verify(json.loads((OUT / 'frozen-source-hashes.json').read_text())),
               'installed_sources': verify(pre['package_hashes'], True), 'tests': tests,
               'port_closed': not port_open(config('hybrid')),
               'compute_processes': subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv'],
                                                  capture_output=True, text=True).stdout,
               'result': 'stopped_before_measurement', 'gpu_retry': False, 'offline_repair_gpu_verified': False}
    write_json(OUT / 'closure.json', closure)
    assert all(closure[k]['pass'] for k in ('old_evidence', 'frozen_sources', 'installed_sources'))
    assert all(t['returncode'] == 0 for t in tests)
    assert closure['port_closed']
    manifest = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in OUT.rglob('*') if p.is_file() and p.name != 'sha256-manifest.json'}
    write_json(OUT / 'sha256-manifest.json', manifest)
    print(json.dumps(closure, indent=2))


if __name__ == '__main__': main()
