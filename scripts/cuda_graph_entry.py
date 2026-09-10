#!/home/lunacia/.venvs/inference-vllm/bin/python
import json
import os
from pathlib import Path
import sys


def configure(args, mode):
    args = list(args)
    if mode not in ('graph', 'eager'):
        raise ValueError(mode)
    if mode == 'graph':
        args.remove('--enforce-eager')
    args.extend(['--compilation-config', json.dumps({'mode': 0,
        'cudagraph_mode': 'FULL_DECODE_ONLY' if mode == 'graph' else 'NONE',
        'cudagraph_capture_sizes': [1, 2, 4], 'max_cudagraph_capture_size': 4}),
        '--cudagraph-metrics', '--enable-prompt-tokens-details'])
    return args


if __name__ == '__main__':
    sys.argv = configure(sys.argv, os.environ['CUDA_GRAPH_MODE'])
    Path(os.environ['CUDA_GRAPH_LAUNCH']).write_text(json.dumps({'argv': sys.argv,
        'environment': {k: os.environ.get(k) for k in ('VLLM_PLUGINS', 'VLLM_USE_V2_MODEL_RUNNER',
        'VLLM_USE_FLASHINFER_SAMPLER', 'C_INCLUDE_PATH')}}, indent=2))
    from vllm.entrypoints.cli.main import main
    main()
