"""Verify retained evidence and cleanup after the bounded GPU session."""
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/stage2-bandwidth'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old = json.loads((ROOT/'artifacts/stage2/evidence-sha256.json').read_text())
    failures = [p for p,h in old.items() if digest(ROOT/'artifacts/stage2'/p) != h]
    stage1 = [line.split(maxsplit=1) for line in (ROOT/'artifacts/stage2/stage1-before.sha256').read_text().splitlines()]
    failures += [p for h,p in stage1 if digest(ROOT/p) != h]
    assert not failures, failures
    parse = runpy.run_path(str(ROOT/'scripts/analyze-bandwidth-check.py'))['parse']
    assert parse(OUT/'capture-1/counters-export.csv') == parse(OUT/'capture-1/counters-installed-export.csv')
    analysis = json.loads((OUT/'analysis.json').read_text())
    assert len(analysis['captures']) == 3 and analysis['total_kernels'] == 24
    for c in analysis['captures']:
        assert c['request_count'] == c['successful_requests'] == 10 and c['fixed_lengths_valid']
        assert len(c['kernels']) == 8
    listeners = [list(c.laddr) for c in psutil.net_connections() if c.status == 'LISTEN' and c.laddr.port == 8000]
    processes = [{'pid':p.pid,'name':p.name()} for p in psutil.process_iter() if 'vllm' in p.name().lower() or p.name().lower() == 'ncu']
    gpu = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
    assert not listeners and not processes and not gpu, (listeners, processes, gpu)
    # Charge an explicit two-minute allowance for tiny permission probes and cleanup,
    # not the user's offline wait or read-only report interpretation.
    charged = analysis['capture_wall_minutes'] + 2
    previous = json.loads((ROOT/'artifacts/stage2/budget-final.json').read_text())['conservative_elapsed_minutes']
    assert charged <= 20 and previous + charged <= 90
    result = {'passed': True, 'stage1_hashes_verified': len(stage1), 'stage2_hashes_verified': len(old),
              'independent_windows_linux_export_equal': True, 'requests_passed': 30, 'kernels': 24,
              'port_8000_listeners': listeners, 'vllm_ncu_processes': processes, 'gpu_compute_processes': gpu,
              'capture_wall_minutes': analysis['capture_wall_minutes'], 'probe_cleanup_allowance_minutes': 2,
              'charged_supplement_minutes': charged, 'charged_combined_minutes': previous+charged,
              'cloud_spend_aud': 0,
              'source_sha256': {str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'scripts').glob('*bandwidth*.py'))}}
    (OUT/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
