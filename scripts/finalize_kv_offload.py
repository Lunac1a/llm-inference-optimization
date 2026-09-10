"""CPU-only closure for native offload attempt; never launches model or probe."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from stage3_lib import ROOT, write_json
from finalize_mixed_load import verify
from local_document_qa import configuration, port_open

OUT = ROOT / 'artifacts/kv-offload'


def main():
    pre = json.loads((OUT / 'preflight.json').read_text())
    tests = []
    for pattern in ('test_kv_offload.py', 'test_mixed_load.py', 'test_hybrid_prefill.py'):
        r = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests'), '-p', pattern],
                           capture_output=True, text=True, timeout=60)
        (OUT / (pattern + '.log')).write_text(r.stdout + r.stderr)
        tests.append({'pattern': pattern, 'returncode': r.returncode})
    closed = {'old_evidence': verify(json.loads((OUT / 'old-evidence-hashes.json').read_text())),
              'frozen_sources': verify(json.loads((OUT / 'frozen-source-hashes.json').read_text())),
              'installed_sources': verify(pre['package_hashes'], True), 'tests': tests,
              'port_closed': not port_open(configuration()),
              'compute_processes': subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv'],
                                                 capture_output=True, text=True).stdout,
              'result': 'compatibility_gate_failed_before_model', 'model_requests': 0,
              'performance_claim': False, 'gpu_retry': False}
    write_json(OUT / 'closure.json', closed)
    assert all(closed[k]['pass'] for k in ('old_evidence', 'frozen_sources', 'installed_sources'))
    assert all(t['returncode'] == 0 for t in tests)
    assert closed['port_closed'] and closed['compute_processes'].strip() == 'pid, process_name'
    write_json(OUT / 'sha256-manifest.json', {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in OUT.rglob('*') if p.is_file() and p.name != 'sha256-manifest.json'})
    print(json.dumps(closed, indent=2))


if __name__ == '__main__': main()
