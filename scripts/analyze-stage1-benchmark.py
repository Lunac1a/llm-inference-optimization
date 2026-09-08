"""Summarize vLLM bench serve detailed JSON without discarding raw results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
import statistics
from datetime import datetime, timezone


def first_number(data: dict, keys: tuple[str, ...]):
    for key in keys:
        value = data.get(key)
        if type(value) in (int, float) and math.isfinite(value):
            return float(value)
    return None


def summarize_file(path: pathlib.Path, expected_requests=32, expected_output=128):
    data = json.loads(path.read_text(encoding="utf-8"))
    match = re.fullmatch(r"round(\d+)-input(\d+)-concurrency(\d+)", path.stem)
    if not match:
        raise ValueError(f"Unexpected result filename: {path.name}")
    round_number, input_tokens, concurrency = (int(value) for value in match.groups())
    summary = {
        "file": path.name,
        "round": round_number,
        "input_tokens": input_tokens,
        "concurrency": concurrency,
        "completed": data.get("completed", data.get("num_completed_requests")),
        "failed": data.get("failed", data.get("num_failed_requests")),
        "request_throughput_req_s": first_number(data, ("request_throughput", "request_throughput_rps")),
        "output_throughput_tok_s": first_number(data, ("output_throughput", "output_throughput_tok_s")),
        "total_token_throughput_tok_s": first_number(data, ("total_token_throughput",)),
        "mean_ttft_ms": first_number(data, ("mean_ttft_ms", "mean_ttft")),
        "mean_tpot_ms": first_number(data, ("mean_tpot_ms", "mean_tpot")),
        "mean_e2el_ms": first_number(data, ("mean_e2el_ms", "mean_e2el")),
        "p50_ttft_ms": first_number(data, ("p50_ttft_ms",)),
        "p95_ttft_ms": first_number(data, ("p95_ttft_ms",)),
        "p50_tpot_ms": first_number(data, ("p50_tpot_ms",)),
        "p95_tpot_ms": first_number(data, ("p95_tpot_ms",)),
        "p50_e2el_ms": first_number(data, ("p50_e2el_ms",)),
        "p95_e2el_ms": first_number(data, ("p95_e2el_ms",)),
    }
    for metric in ("ttft", "tpot", "e2el"):
        percentiles = data.get(f"{metric}_percentiles_ms") or data.get(f"{metric}_percentiles") or {}
        if isinstance(percentiles, dict):
            for percentile in (50, 95):
                value = percentiles.get(str(percentile), percentiles.get(percentile))
                if isinstance(value, (int, float)):
                    summary[f"p{percentile}_{metric}_ms"] = float(value)
    # Pinned vLLM writes parallel arrays. Missing evidence is unknown, not zero.
    errors = data.get("errors")
    valid_errors = isinstance(errors, list) and all(isinstance(e, str) for e in errors)
    summary["detailed_request_count"] = len(errors) if valid_errors else None
    summary["detailed_error_count"] = sum(bool(e) for e in errors) if valid_errors else None
    summary["request_evidence_passed"] = (
        valid_errors and len(errors) == expected_requests and not any(errors)
        and data.get("num_prompts") == expected_requests
        and summary["completed"] == expected_requests and summary["failed"] == 0
        and data.get("max_concurrency") == concurrency
        and data.get("input_lens") == [input_tokens] * expected_requests
        and data.get("output_lens") == [expected_output] * expected_requests
        and all(isinstance(data.get(key), list) and len(data[key]) == expected_requests
                for key in ("ttfts", "itls", "generated_texts"))
        and all(summary[f"p{p}_{metric}_ms"] is not None
                for p in (50, 95) for metric in ("ttft", "tpot", "e2el"))
    )
    return summary


EXPECTED_GROUPS = {(512, 1), (512, 4), (2048, 1), (2048, 4)}


def load_run(run_dir):
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("rounds") != 3 or protocol.get("measured_requests") != 32 or protocol.get("output_length") != 128:
        raise ValueError("Expected fixed Stage 1 protocol: 3 rounds, 32 requests, 128 output tokens")
    selected = protocol.get("only_groups", "")
    groups = {tuple(map(int, g.split(":"))) for g in selected.split(",")} if selected else EXPECTED_GROUPS
    if not groups or not groups <= EXPECTED_GROUPS:
        raise ValueError("Unexpected configuration groups")
    rows = [summarize_file(p) for p in sorted(run_dir.glob("round*-input*-concurrency*.json"))]
    expected = {(r, i, c) for r in (1, 2, 3) for i, c in groups}
    actual = {(row["round"], row["input_tokens"], row["concurrency"]) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("Missing, duplicate, or unexpected benchmark rounds")
    return protocol, rows


def stability_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["input_tokens"], row["concurrency"]), []).append(row)
    result = []
    for (input_tokens, concurrency), group in sorted(grouped.items()):
        values = [row["output_throughput_tok_s"] for row in group]
        valid = (len(group) == 3 and {row["round"] for row in group} == {1, 2, 3}
                 and all(v is not None and v > 0 for v in values))
        mean = statistics.mean(values) if valid else None
        cv = statistics.stdev(values) / mean * 100 if valid else None
        result.append({"input_tokens": input_tokens, "concurrency": concurrency,
                       "rounds": len(group), "output_throughput_mean_tok_s": mean,
                       "output_throughput_cv_percent": cv,
                       "stability_passed_cv_le_5_percent": cv is not None and cv <= 5})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    run_dir = pathlib.Path(args.run_dir)
    protocol, rows = load_run(run_dir)
    fields = list(rows[0])
    summary_path = run_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    cv_rows = stability_rows(rows)
    cv_path = run_dir / "stability.csv"
    with cv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cv_rows[0]))
        writer.writeheader()
        writer.writerows(cv_rows)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "raw_result_count": len(rows),
        "rows": rows,
        "stability": cv_rows,
        "formal_stability_passed": all(row["stability_passed_cv_le_5_percent"] for row in cv_rows),
        "all_requests_passed": all(row["request_evidence_passed"] for row in rows),
        "parser_note": "Raw vLLM JSON remains authoritative; fields absent in this vLLM release are recorded as null.",
    }
    (run_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "raw_result_count": len(rows), "formal_stability_passed": report["formal_stability_passed"]}, indent=2))
    return 0 if report["all_requests_passed"] and report["formal_stability_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
