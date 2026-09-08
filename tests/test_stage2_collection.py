"""Parameterized evidence validation must not weaken Stage 1 defaults."""
import json
from pathlib import Path
import runpy
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
summarize = runpy.run_path(str(ROOT / 'scripts/analyze-stage1-benchmark.py'))['summarize_file']


class Stage2Tests(unittest.TestCase):
    def test_pinned_cli_skips_probe_and_checks_warmup_errors(self):
        valid = runpy.run_path(str(ROOT/'scripts/stage2-run.py'))['requests_valid']
        requests = [dict(success=True,error='',prompt_len=512,output_tokens=128) for _ in range(20)]
        self.assertTrue(valid(requests,16,512,128))
        self.assertFalse(valid(requests[:-1],16,512,128))
        requests[0]['error']='warmup failure'
        self.assertFalse(valid(requests,16,512,128))

    def test_variable_lengths_and_count_fail_closed(self):
        source = ROOT / 'artifacts/stage1/benchmarks/20260907T140950Z/round1-input512-concurrency1.json'
        raw = json.loads(source.read_text())
        for key in ('errors', 'input_lens', 'ttfts', 'itls', 'generated_texts'):
            raw[key] = raw[key][:16]
        raw.update(num_prompts=16, completed=16, failed=0, output_lens=[512]*16)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / source.name
            path.write_text(json.dumps(raw))
            self.assertTrue(summarize(path,16,512)['request_evidence_passed'])
            self.assertFalse(summarize(path)['request_evidence_passed'])
            raw['output_lens'][0] = 511
            path.write_text(json.dumps(raw))
            self.assertFalse(summarize(path,16,512)['request_evidence_passed'])
            raw['output_lens'][0] = 512
            raw['errors'][0] = 'timeout'
            path.write_text(json.dumps(raw))
            self.assertFalse(summarize(path,16,512)['request_evidence_passed'])
