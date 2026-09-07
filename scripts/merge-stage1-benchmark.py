"""Merge the final Stage 1 rows after targeted stability reruns."""

from __future__ import annotations

import argparse
import json
import pathlib
from datetime import datetime, timezone


EXPECTED_GROUPS = {(512, 1), (512, 4), (2048, 1), (2048, 4)}


def load_summary(run_dir: pathlib.Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def row_key(row: dict) -> tuple[int, int]:
    return int(row["input_tokens"]), int(row["concurrency"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_run", type=pathlib.Path)
    parser.add_argument("replacement_runs", nargs="+", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    rows: dict[tuple[int, int], dict] = {}
    stability: dict[tuple[int, int], dict] = {}
    sources: dict[tuple[int, int], str] = {}
    run_dirs = [args.base_run, *args.replacement_runs]
    for run_dir in run_dirs:
        report = load_summary(run_dir)
        for row in report["rows"]:
            key = row_key(row)
            rows[key] = row
            sources[key] = str(run_dir)
        for item in report["stability"]:
            stability[row_key(item)] = item

    missing = EXPECTED_GROUPS - rows.keys()
    missing_stability = EXPECTED_GROUPS - stability.keys()
    if missing or missing_stability:
        raise SystemExit(f"Missing final groups: rows={sorted(missing)}, stability={sorted(missing_stability)}")

    ordered_keys = sorted(EXPECTED_GROUPS)
    selected_rows = []
    for key in ordered_keys:
        row = dict(rows[key])
        row["source_run"] = sources[key]
        selected_rows.append(row)

    all_requests_passed = all(
        row.get("completed") == 32 and row.get("failed") == 0 and row.get("detailed_error_count") == 0
        for row in selected_rows
    )
    stability_rows = [stability[key] for key in ordered_keys]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_run": str(args.base_run),
        "replacement_runs": [str(path) for path in args.replacement_runs],
        "rows": selected_rows,
        "stability": stability_rows,
        "all_requests_passed": all_requests_passed,
        "formal_stability_passed": all(
            item["stability_passed_cv_le_5_percent"] for item in stability_rows
        ),
        "raw_evidence_note": "Each selected row remains available in its source run directory; targeted reruns replace only the failed configuration groups.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "all_requests_passed": report["all_requests_passed"],
        "formal_stability_passed": report["formal_stability_passed"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
