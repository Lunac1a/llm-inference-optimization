"""Guard against labelling a GPU hit as CPU restoration."""
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from validate_kv_offload import controls


class OffloadTests(unittest.TestCase):
    def test_restoration_requires_external_hits_and_transfer(self):
        def row(name, value, **labels): return {'name': name, 'delta': value, 'labels': labels}
        local = row('vllm:prefix_cache_hits_total', 16416)
        external = row('vllm:external_prefix_cache_hits_total', 16416)
        transfer = row('vllm:kv_offload_total_bytes_total', 2**30, transfer_type='CPU_to_GPU')
        self.assertFalse(controls([local], 'offload', 'revisit', 16446)['pass'])
        self.assertFalse(controls([external], 'offload', 'revisit', 16446)['pass'])
        self.assertTrue(controls([external, transfer], 'offload', 'revisit', 16446)['pass'])
        self.assertFalse(controls([local], 'baseline', 'revisit', 16446)['pass'])

    def test_real_prometheus_counter_name(self):
        from prometheus_client import Counter, CollectorRegistry, generate_latest
        registry = CollectorRegistry()
        Counter('vllm:kv_offload_total_bytes', 'bytes', registry=registry).inc()
        self.assertIn(b'vllm:kv_offload_total_bytes_total ', generate_latest(registry))

    def test_launcher_changes_only_kv_offload_flags(self):
        cli = ModuleType('vllm.entrypoints.cli.main')
        cli.main = lambda: None
        root = Path(__file__).resolve().parents[1]
        for mode in ('baseline', 'offload'):
            with tempfile.TemporaryDirectory() as d:
                file = Path(d) / 'launch.json'
                with patch.dict(os.environ, {'KV_OFFLOAD_MODE': mode, 'KV_OFFLOAD_LAUNCH': str(file)}), \
                     patch.dict(sys.modules, {'vllm.entrypoints.cli.main': cli}), patch.object(sys, 'argv', ['entry.py']):
                    runpy.run_path(str(root / 'scripts/kv_offload_entry.py'), run_name='__main__')
                args = json.loads(file.read_text())['argv']
                self.assertEqual('--kv-offloading-size' in args, mode == 'offload')
                self.assertNotIn('--cpu-offload-gb', args)


if __name__ == '__main__': unittest.main()
