#!/home/lunacia/.venvs/inference-vllm/bin/python
"""Experiment-only CLI adapter; preserve shared historical launcher unchanged."""
import json
import os
from pathlib import Path
import sys

if __name__=='__main__':
    if '--enable-chunked-prefill' not in sys.argv:
        raise SystemExit('Expected historical launcher argument')
    sys.argv.remove('--enable-chunked-prefill')
    sys.argv.append('--no-enable-chunked-prefill')
    evidence=Path(os.environ['HYBRID_LAUNCH_EVIDENCE'])
    evidence.write_text(json.dumps({'argv':sys.argv,'environment':{k:os.environ.get(k) for k in
        ['PYTHONPATH','VLLM_PLUGINS','VLLM_USE_V2_MODEL_RUNNER','VLLM_USE_FLASHINFER_SAMPLER']}},indent=2)+'\n')
    from vllm.entrypoints.cli.main import main
    main()
