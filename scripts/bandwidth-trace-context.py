"""Read-only context from the original Stage 2 low-concurrency trace."""
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = next((ROOT/'artifacts/stage2/traces/profile-low').glob('rank*.gz'))
events = json.load(gzip.open(path))['traceEvents']
mm = {e['args']['External id']: e for e in events if e.get('name') == 'aten::mm'}
kernels = sorted((e for e in events if e.get('cat') == 'kernel'), key=lambda e:e['ts'])
gemv = [e for e in kernels if 'cublasGemv' in e['name']]
samples = []
for k in gemv[:8]:
    op = mm.get(k['args']['External id'])
    samples.append({'grid': k['args']['grid'], 'block': k['args']['block'],
                    'duration_us': k['dur'], 'mm_input_dims': op['args']['Input Dims'] if op else None})
windows = [e for e in events if e.get('cat') == 'gpu_user_annotation' and e['name'].startswith('execute_context_0(0)')]
decode = [k for k in kernels if any(w['ts'] <= k['ts'] and k['ts'] + k['dur'] <= w['ts'] + w['dur'] + .001 for w in windows)]
gemv_us = sum(k['dur'] for k in decode if 'cublasGemv' in k['name'])
kernel_us = sum(k['dur'] for k in decode)
result = {'source': str(path.relative_to(ROOT)), 'first_eight_gemv': samples,
          'decode_windows': len(windows), 'decode_kernel_count': len(decode),
          'decode_gemv_summed_us': gemv_us, 'decode_all_kernels_summed_us': kernel_us,
          'gemv_fraction_of_summed_decode_kernel_time': gemv_us/kernel_us,
          'limits': 'Historical cap4 Torch-profiled trace. Shapes/order/grid support inference about new cap8 samples; no direct cross-run correlation or unprofiled latency share.'}
(ROOT/'artifacts/stage2-bandwidth/trace-context.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
