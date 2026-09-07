"""Recompute complete three-round groups from raw evidence; audit rerun policy."""
from __future__ import annotations

import argparse
import json
import pathlib
import runpy
from datetime import datetime, timezone

helpers = runpy.run_path(str(pathlib.Path(__file__).with_name("analyze-stage1-benchmark.py")))
load_run = helpers["load_run"]
stability_rows = helpers["stability_rows"]
EXPECTED_GROUPS = helpers["EXPECTED_GROUPS"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_run", type=pathlib.Path)
    parser.add_argument("replacement_runs", nargs="*", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    groups = {}
    deviations = []
    base_protocol = None
    previous_dir = None
    run_dirs = [args.base_run, *args.replacement_runs]
    if len({p.resolve() for p in run_dirs}) != len(run_dirs):
        raise ValueError("Duplicate run directories")
    if len(args.replacement_runs) > 1:
        deviations.append("More than one rerun was selected; the original protocol allows one.")
    fixed = ("model_id", "model_revision", "served_model_name", "dataset", "request_rate",
             "warmups", "measured_requests", "rounds", "seed", "input_lengths", "concurrency", "output_length")
    for index, run_dir in enumerate(run_dirs):
        protocol, rows = load_run(run_dir)
        if index == 0:
            base_protocol = protocol
            if protocol.get("rerun_of"):
                deviations.append("Base selection is itself a rerun; original-run provenance requires review.")
        else:
            if any(k not in protocol or protocol[k] != base_protocol.get(k) for k in fixed):
                raise ValueError("Cannot combine different benchmark protocols")
            parent = protocol.get("rerun_of")
            if not parent or pathlib.Path(parent).resolve() != previous_dir.resolve():
                deviations.append(f"Rerun parent is not the preceding run: {run_dir}")
        incoming = {}
        for row in rows:
            key = (row["input_tokens"], row["concurrency"])
            incoming.setdefault(key, []).append({**row, "source_run": str(run_dir)})
        if index == 0 and set(incoming) != EXPECTED_GROUPS:
            raise ValueError("Base run must contain all four complete configuration groups")
        full_rerun = index == 1 and set(incoming) == EXPECTED_GROUPS
        previous_stable = index and all(s["stability_passed_cv_le_5_percent"]
                                       for s in stability_rows([r for g in groups.values() for r in g]))
        if full_rerun and previous_stable:
            deviations.append("Full rerun replaced an already stable baseline")
        for key, group in incoming.items():
            if index and not full_rerun and stability_rows(groups[key])[0]["stability_passed_cv_le_5_percent"]:
                deviations.append(f"Replacement of a group that already met stability: {key}")
            # Replace the entire group, never just its last round.
            groups[key] = group
        previous_dir = run_dir
    selected_rows = [row for key in sorted(groups) for row in sorted(groups[key], key=lambda r: r["round"])]
    stability = stability_rows(selected_rows)
    requests_passed = all(row["request_evidence_passed"] for row in selected_rows)
    stable = all(row["stability_passed_cv_le_5_percent"] for row in stability)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_run": str(args.base_run),
        "replacement_runs": [str(p) for p in args.replacement_runs],
        "rows": selected_rows,
        "raw_result_count": len(selected_rows),
        "stability": stability,
        "all_requests_passed": requests_passed,
        "formal_stability_passed": stable,
        "protocol_compliant": not deviations,
        "protocol_deviations": deviations,
        "benchmark_acceptance_passed": requests_passed and stable and not deviations,
        "stage1_passed": False,
        "stage1_note": "Benchmark-only analysis cannot certify API acceptance or restart verification.",
        "raw_evidence_note": "Recomputed from raw JSON. Full groups retain all three rounds; CV alone does not override protocol deviations.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("raw_result_count", "all_requests_passed", "formal_stability_passed", "protocol_compliant", "benchmark_acceptance_passed")}, indent=2))
    return 0 if report["benchmark_acceptance_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
