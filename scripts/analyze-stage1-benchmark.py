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
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)
    return None


def summarize_file(path: pathlib.Path):
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
    details = data.get("requests") or data.get("request_results") or data.get("per_request") or []
    if isinstance(details, list):
        summary["detailed_request_count"] = len(details)
        summary["detailed_error_count"] = sum(bool(item.get("error")) for item in details if isinstance(item, dict))
    else:
        errors = data.get("errors")
        summary["detailed_request_count"] = len(errors) if isinstance(errors, list) else None
        summary["detailed_error_count"] = sum(bool(error) for error in errors) if isinstance(errors, list) else None
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    run_dir = pathlib.Path(args.run_dir)
    files = sorted(run_dir.glob("round*-input*-concurrency*.json"))
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    only_groups = [group for group in protocol.get("only_groups", "").split(",") if group]
    expected_file_count = 12 if not only_groups else len(only_groups) * int(protocol.get("rounds", 3))
    if len(files) != expected_file_count:
        raise SystemExit(f"Expected {expected_file_count} raw benchmark JSON files, found {len(files)}")
    rows = [summarize_file(path) for path in files]
    fields = list(rows[0])
    summary_path = run_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for row in rows:
        key = (row["input_tokens"], row["concurrency"])
        grouped.setdefault(key, []).append(row)
    cv_rows = []
    for (input_tokens, concurrency), group in sorted(grouped.items()):
        values = [row["output_throughput_tok_s"] for row in group if row["output_throughput_tok_s"] is not None]
        mean = statistics.mean(values) if values else None
        cv_percent = (statistics.stdev(values) / mean * 100) if len(values) > 1 and mean else None
        cv_rows.append({
            "input_tokens": input_tokens,
            "concurrency": concurrency,
            "rounds": len(group),
            "output_throughput_mean_tok_s": mean,
            "output_throughput_cv_percent": cv_percent,
            "stability_passed_cv_le_5_percent": cv_percent is not None and cv_percent <= 5.0,
        })
    cv_path = run_dir / "stability.csv"
    with cv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cv_rows[0]))
        writer.writeheader()
        writer.writerows(cv_rows)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "raw_result_count": len(files),
        "rows": rows,
        "stability": cv_rows,
        "formal_stability_passed": all(row["stability_passed_cv_le_5_percent"] for row in cv_rows),
        "parser_note": "Raw vLLM JSON remains authoritative; fields absent in this vLLM release are recorded as null.",
    }
    (run_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "raw_result_count": len(files), "formal_stability_passed": report["formal_stability_passed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
