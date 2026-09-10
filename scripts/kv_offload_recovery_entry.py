#!/home/lunacia/.venvs/inference-vllm/bin/python
import json
import os
from pathlib import Path
import sys

if __name__ == '__main__':
    mode = os.environ['KV_OFFLOAD_MODE']
    if mode == 'offload':
        assert os.environ['VLLM_PLUGINS'] == 'inference_kv_transport'
        sys.argv.extend(['--kv-offloading-size', '8', '--kv-offloading-backend', 'native'])
    elif mode != 'baseline':
        raise ValueError(mode)
    sys.argv.append('--enable-prompt-tokens-details')
    Path(os.environ['KV_OFFLOAD_LAUNCH']).write_text(json.dumps({'argv': sys.argv,
        'environment': {k: os.environ.get(k) for k in ('VLLM_PLUGINS', 'PYTHONPATH', 'VLLM_USE_SIMPLE_KV_OFFLOAD',
                          'VLLM_USE_V2_MODEL_RUNNER', 'VLLM_USE_FLASHINFER_SAMPLER', 'C_INCLUDE_PATH')}}, indent=2))
    from vllm.entrypoints.cli.main import main
    main()
