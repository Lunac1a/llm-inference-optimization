"""Fresh user-authorized execution; original run remains immutable."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from unittest.mock import AsyncMock, patch

import validate_mixed_load as collector
from stage3_lib import ROOT, write_json
from validate_fp8_capacity import hashes

OUT = ROOT / 'artifacts/mixed-load-rerun'
collector.OUT = OUT


def prepare():
    # Verify real installed route registration with no server/model/CUDA startup.
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from vllm.entrypoints.serve.dev.cache.api_router import attach_router
    app = FastAPI()
    app.state.engine_client = AsyncMock()
    attach_router(app)
    with TestClient(app) as client:
        response = client.post('/reset_prefix_cache', json={})
    assert response.status_code == 200
    app.state.engine_client.reset_prefix_cache.assert_awaited_once_with(False, False)
    collector.prepare()
    write_json(OUT / 'reset-api-offline.json', {'status': response.status_code,
        'mock_engine_reset_awaited': True, 'gpu_started': False,
        'limitation': 'Handler contract only; real reset/cache validation required in trials'})


def execute():
    if (OUT / 'execution.json').exists():
        raise RuntimeError('Refuse execution overwrite')
    write_json(OUT / 'execution.json', {'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'sources': hashes([Path(__file__), Path(collector.__file__), ROOT / 'scripts/mixed_vllm_entry.py',
                           ROOT / 'docs/mixed-load-plan.md', ROOT / 'docs/mixed-load-rerun-plan.md']),
        'fresh_budget_minutes': 30, 'old_evidence_directory': 'artifacts/mixed-load'})
    asyncio.run(collector.collect())


def analyze():
    import analyze_mixed_load
    analyze_mixed_load.OUT = OUT
    analyze_mixed_load.analyze()


def close():
    import finalize_mixed_load as old
    pre = json.loads((OUT / 'preflight.json').read_text())
    tests = []
    for pattern in ('test_mixed_load.py', 'test_hybrid_prefill.py'):
        p = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests'), '-p', pattern],
                           capture_output=True, text=True, timeout=60)
        (OUT / (pattern + '.log')).write_text(p.stdout + p.stderr)
        tests.append({'pattern': pattern, 'returncode': p.returncode})
    cleanup = json.loads((OUT / 'cleanup.json').read_text())
    result = {'old_evidence': old.verify(json.loads((OUT / 'old-evidence-hashes.json').read_text())),
              'frozen_sources': old.verify(json.loads((OUT / 'frozen-source-hashes.json').read_text())),
              'installed_sources': old.verify(pre['package_hashes'], True), 'tests': tests,
              'cleanup': cleanup, 'failure': (OUT / 'failure.json').exists()}
    write_json(OUT / 'closure.json', result)
    assert all(result[k]['pass'] for k in ('old_evidence', 'frozen_sources', 'installed_sources'))
    assert all(t['returncode'] == 0 for t in tests)
    assert cleanup['port_closed'] and cleanup['compute_processes'].strip() == 'pid, process_name'
    write_json(OUT / 'sha256-manifest.json', {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in OUT.rglob('*') if p.is_file() and p.name != 'sha256-manifest.json'})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'run', 'analyze', 'close'])
    {'prepare': prepare, 'run': execute, 'analyze': analyze, 'close': close}[parser.parse_args().phase]()
