"""Regression checks for the acceptance errors found during review."""
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
API = runpy.run_path(str(ROOT / "scripts/check-vllm-api.py"))
BENCH = runpy.run_path(str(ROOT / "scripts/analyze-stage1-benchmark.py"))


class AcceptanceTests(unittest.TestCase):
    def test_normal_answer_requires_stop_and_no_reasoning(self):
        check = API["complete_answer"]
        self.assertTrue(check("323", "stop"))
        self.assertFalse(check("323", "length"))
        self.assertFalse(check("<think>working", "stop"))
        self.assertFalse(check("323", "stop", "hidden reasoning"))

    def test_stream_requires_clean_terminal_event(self):
        check = API["stream_answer_passed"]
        terminal = [{"choices": [{"finish_reason": "stop"}]}]
        self.assertTrue(check(200, "hello", terminal, True))
        self.assertFalse(check(200, "hello", terminal, False))
        self.assertFalse(check(200, "hello", [], True))
        self.assertFalse(check(200, "hello", terminal + [{"transport_error": "lost"}], True))

    def test_raw_error_array_is_required(self):
        source = ROOT / "artifacts/stage1/benchmarks/20260907T140950Z/round1-input512-concurrency1.json"
        raw = json.loads(source.read_text())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / source.name
            for errors, expected_count, expected_pass in [([""] * 32, 32, True),
                                                        (["failure"] + [""] * 31, 32, False),
                                                        (None, None, False)]:
                raw["errors"] = errors
                path.write_text(json.dumps(raw))
                result = BENCH["summarize_file"](path)
                self.assertEqual(result["detailed_request_count"], expected_count)
                self.assertEqual(result["request_evidence_passed"], expected_pass)

    def test_missing_rounds_are_not_stable(self):
        rows = [{"round": 1, "input_tokens": 512, "concurrency": 1,
                 "output_throughput_tok_s": 10}] * 3
        self.assertFalse(BENCH["stability_rows"](rows)[0]["stability_passed_cv_le_5_percent"])

    def test_merge_preserves_twelve_rows_and_rejects_excess_reruns(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "report.json"
            result = subprocess.run([sys.executable, str(ROOT / "scripts/merge-stage1-benchmark.py"),
                "artifacts/stage1/benchmarks/20260907T115614Z",
                "artifacts/stage1/benchmarks/20260907T135907Z",
                "artifacts/stage1/benchmarks/20260907T140950Z", "--output", str(output)],
                cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            data = json.loads(output.read_text())
            self.assertEqual(len(data["rows"]), 12)
            self.assertEqual(len({(r["input_tokens"], r["concurrency"], r["round"]) for r in data["rows"]}), 12)
            self.assertFalse(data["protocol_compliant"])
            self.assertFalse(data["benchmark_acceptance_passed"])


if __name__ == "__main__":
    unittest.main()
