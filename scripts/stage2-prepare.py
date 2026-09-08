"""Read-only installed-source inspection and pre-collection manifest."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

root = Path(__file__).resolve().parents[1]
out = root / 'artifacts/stage2'
package = Path(importlib.util.find_spec('vllm').origin).parent
files = ['benchmarks/serve.py', 'benchmarks/lib/endpoint_request_func.py',
         'config/profiler.py', 'v1/metrics/loggers.py', 'v1/core/sched/scheduler.py',
         'v1/worker/gpu_model_runner.py']
source = out / 'installed-source'
source.mkdir(exist_ok=True)
for name in files:
    (source / name.replace('/', '__')).write_bytes((package / name).read_bytes())
paths = list((root / 'scripts').glob('*.*')) + list((root / 'configs').glob('*.env')) + list((root / 'locks').glob('*')) + list(source.glob('*'))
(out / 'frozen-sha256.json').write_text(json.dumps({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}, indent=2))
(out / 'protocol.json').write_text(json.dumps({
    'seed': 42, 'temperature': 0, 'ignore_eos': True, 'range_ratio': 0, 'prefix': 0,
    'request_timeout_seconds': 120, 'batch_timeout_seconds': 600,
    'anchor': {'workload': [512,128,4], 'batches': 2, 'requests': 16, 'warmups': 4, 'max_difference_percent':5},
    'sweep_order': [[w,i,o,c] for w,i,o in [('A',512,128),('B',2048,128),('C',512,512)] for c in [1,2,4,8,16]],
    'sweep_requests':16, 'warmups':4, 'confirmation_requests':32, 'confirmation_rounds':3,
    'confirmation_rotation':'[low,mid,high], [mid,high,low], [high,low,mid]',
    'max_complete_confirmation_repeat':1,
    'cap_contrast':'three paired rounds, cap order 4/8, 8/4, 4/8; concurrency 8',
    'knee':'doubling concurrency gains <10% output throughput and increases p95 TTFT or E2E >25%',
    'stop':'OOM/crash/error/invalid lengths/timeout; anchor drift >5%; repeated confirmation CV >5%; 90-minute budget',
    'profiler':'selected low and knee, 2-4 requests; separate restart with torch profiler',
    'upstream_probe':'ready_check_timeout_sec defaults to 0, so no probe is sent; four warmups plus measured requests are retained in requests.jsonl',
}, indent=2))
print('Manifest frozen. Start budget immediately before server startup.')
