#!/home/lunacia/.venvs/inference-vllm/bin/python
"""Scoped launch adapter; never changes the historical hybrid launcher."""
import json
import os
from pathlib import Path
import sys

if __name__ == '__main__':
    mode = os.environ['MIXED_MODE']
    if mode == 'full':
        sys.argv.remove('--enable-chunked-prefill')
        sys.argv.append('--no-enable-chunked-prefill')
    elif mode == 'chunk':
        sys.argv.extend(['--long-prefill-token-threshold', '2048'])
    else:
        raise ValueError(mode)
    sys.argv.append('--enable-prompt-tokens-details')
    Path(os.environ['HYBRID_LAUNCH_EVIDENCE']).write_text(json.dumps({
        'argv': sys.argv, 'mode': mode,
        'environment': {k: os.environ.get(k) for k in ['PYTHONPATH', 'VLLM_PLUGINS']}}, indent=2))
    from vllm.entrypoints.cli.main import main
    main()
