"""Optional pinned-runtime CPU KV profile for the existing local API."""
from contextlib import contextmanager
import os
from pathlib import Path
from stage3_lib import ROOT, OwnedVllm


def memory():
    return {line.split(':')[0]: int(line.split()[1]) * 1024
            for line in Path('/proc/meminfo').read_text().splitlines()
            if line.startswith('MemAvailable:')}


@contextmanager
def launch(folder):
    values = {'KV_OFFLOAD_MODE': 'offload', 'KV_OFFLOAD_LAUNCH': str(folder / 'actual-launch.json'),
              'VLLM_PLUGINS': 'inference_kv_transport', 'VLLM_USE_SIMPLE_KV_OFFLOAD': '0',
              'PYTHONPATH': str(ROOT / 'experiments/kv_offload_compat')}
    before = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k, v in before.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class CpuKvServer(OwnedVllm):
    def _paths(self):
        python, _, headers = super()._paths()
        return python, str(ROOT / 'scripts/kv_offload_recovery_entry.py'), headers

    def start(self, timeout=180):
        if not self.candidate['prefix_caching']:
            raise ValueError('CPU KV profile requires prefix caching')
        if memory()['MemAvailable'] < 12 * 2**30:
            raise RuntimeError('CPU KV profile requires at least 12 GiB WSL MemAvailable before startup')
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with launch(self.run_dir):
            return super().start(timeout=timeout)
