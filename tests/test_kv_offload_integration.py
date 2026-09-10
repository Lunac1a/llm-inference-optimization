import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from cpu_kv_api import CpuKvServer, launch
from local_document_qa import configuration
import local_document_qa
from stage3_lib import OwnedVllm
from validate_kv_offload_integration import controls, Server


class IntegrationTests(unittest.TestCase):
    def test_cli_selects_explicit_profile_and_cleans_up(self):
        for cpu in (False, True):
            target = 'cpu_kv_api.CpuKvServer' if cpu else 'local_document_qa.OwnedVllm'
            with patch(target) as factory, patch('local_document_qa.port_open', return_value=False), \
                 patch.object(sys, 'argv', ['local_document_qa.py', 'serve'] + (['--cpu-kv-cache'] if cpu else [])):
                factory.return_value.start.side_effect = KeyboardInterrupt()
                local_document_qa.main()
                factory.assert_called_once()
                factory.return_value.stop.assert_called_once()

    def test_cli_rejects_prefix_off_combination(self):
        with patch('local_document_qa.port_open', return_value=False), \
             patch.object(sys, 'argv', ['local_document_qa.py', 'serve', '--cpu-kv-cache', '--no-prefix-cache']):
            with self.assertRaises(SystemExit) as error:
                local_document_qa.main()
            self.assertEqual(error.exception.code, 2)

    def test_partial_eviction(self):
        def row(name, value, **labels): return {'name': 'vllm:' + name, 'delta': value, 'labels': labels}
        local = row('prefix_cache_hits_total', 3360)
        external = row('external_prefix_cache_hits_total', 13072)
        copied = row('kv_offload_total_bytes_total', 1927544832, transfer_type='CPU_to_GPU')
        self.assertTrue(controls([local], 'baseline', 'revisit', 16446)['pass'])
        self.assertFalse(controls([row('prefix_cache_hits_total', 16432)], 'baseline', 'revisit', 16446)['pass'])
        self.assertTrue(controls([local, external, copied], 'offload', 'revisit', 16446)['pass'])
        self.assertFalse(controls([local, external], 'offload', 'revisit', 16446)['pass'])

    def test_environment_restored_after_failure(self):
        before = dict(os.environ)
        with self.assertRaises(RuntimeError):
            with launch(Path('/tmp/example')):
                self.assertEqual(os.environ['VLLM_PLUGINS'], 'inference_kv_transport')
                raise RuntimeError()
        self.assertEqual(dict(os.environ), before)

    def test_memory_and_prefix_guards(self):
        server = CpuKvServer(configuration(False), Path('/tmp/unused'))
        with self.assertRaises(ValueError): server.start()
        server = CpuKvServer(configuration(), Path('/tmp/unused'))
        with patch('cpu_kv_api.memory', return_value={'MemAvailable': 2**30}):
            with self.assertRaises(RuntimeError): server.start()

    def test_delivery_and_collector_launch_paths(self):
        for cls, name in ((CpuKvServer, 'delivery'), (Server, 'r2-offload'), (Server, 'r2-baseline')):
            with tempfile.TemporaryDirectory() as d:
                server = cls(configuration(), Path(d) / name / 'server')
                def start(instance, timeout=180):
                    if name != 'r2-baseline':
                        self.assertEqual(os.environ['KV_OFFLOAD_MODE'], 'offload')
                    return {'status': 'ready'}
                with patch('cpu_kv_api.memory', return_value={'MemAvailable': 14 * 2**30}), \
                     patch.object(OwnedVllm, 'start', start):
                    self.assertEqual(server.start()['status'], 'ready')


if __name__ == '__main__': unittest.main()
