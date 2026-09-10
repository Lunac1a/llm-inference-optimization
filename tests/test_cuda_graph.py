import os
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import json
from cuda_graph_entry import configure
from validate_cuda_graph import launch


class GraphTests(unittest.TestCase):
    def test_graph_disables_only_eager_and_compilation_stays_off(self):
        source = ['entry', 'serve', 'model', '--enforce-eager']
        for mode in ('eager', 'graph'):
            args = configure(source, mode)
            self.assertEqual('--enforce-eager' in args, mode == 'eager')
            cfg = json.loads(args[args.index('--compilation-config') + 1])
            self.assertEqual(cfg['mode'], 0)
            self.assertEqual(cfg['cudagraph_capture_sizes'], [1, 2, 4])
            self.assertEqual(cfg['cudagraph_mode'], 'FULL_DECODE_ONLY' if mode == 'graph' else 'NONE')
            self.assertNotIn('--kv-offloading-size', args)
        self.assertIn('--enforce-eager', source)

    def test_environment_restored(self):
        before = dict(os.environ)
        with self.assertRaises(RuntimeError):
            with launch('graph', Path('/tmp/example')):
                self.assertEqual(os.environ['VLLM_PLUGINS'], '')
                raise RuntimeError()
        self.assertEqual(dict(os.environ), before)


if __name__ == '__main__': unittest.main()
