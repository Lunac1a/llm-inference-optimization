"""Apply the preregistered Stage 3 gates without changing raw evidence."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
from typing import Any

from stage3_lib import OUT, percentile, sha256_bytes, write_json


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def cv(values: list[float]) -> float | None:
    if len(values) < 2 or not mean(values):
        return None
    return statistics.stdev(values) / mean(values) * 100


def cache_capacity(compat: dict[str, Any]) -> int | None:
    labels = compat.get("cache_config") or {}
    try:
        blocks = int(float(labels["num_gpu_blocks"]))
        block_size = int(float(labels["block_size"]))
        return blocks * block_size
    except (KeyError, TypeError, ValueError):
        return None


def round_metrics(rows: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    selected = [row for row in rows if row["candidate"] == candidate]
    shared = [row["shared"] for row in selected]
    independent = [row["independent"]["requests"] for row in selected]
    shared_throughput = [row["shared"]["including_first_throughput_tok_s"] for row in selected
                         if row["shared"].get("including_first_throughput_tok_s") is not None]
    shared_follow = [row["shared"]["followups"] for row in selected]
    return {
        "candidate": candidate,
        "round_count": len(selected),
        "shared_throughput_values": shared_throughput,
        "shared_throughput_mean": mean(shared_throughput),
        "shared_throughput_cv_percent": cv(shared_throughput),
        "shared_followup_p95_ttft_seconds": mean([row["ttft_seconds"]["p95"] for row in shared_follow if row["ttft_seconds"]["p95"] is not None]),
        "shared_followup_p95_tpot_seconds": mean([row["tpot_seconds"]["p95"] for row in shared_follow if row["tpot_seconds"]["p95"] is not None]),
        "shared_followup_p95_e2e_seconds": mean([row["e2e_seconds"]["p95"] for row in shared_follow if row["e2e_seconds"]["p95"] is not None]),
        "independent_p95_ttft_seconds": mean([row["ttft_seconds"]["p95"] for row in independent if row["ttft_seconds"]["p95"] is not None]),
        "independent_p95_tpot_seconds": mean([row["tpot_seconds"]["p95"] for row in independent if row["tpot_seconds"]["p95"] is not None]),
        "independent_p95_e2e_seconds": mean([row["e2e_seconds"]["p95"] for row in independent if row["e2e_seconds"]["p95"] is not None]),
        "rounds": selected,
    }


def relative_change(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline in (None, 0):
        return None
    return (value / baseline - 1) * 100


def main() -> None:
    compatibility = read(OUT / "compatibility.json")["candidates"]
    quality = read(OUT / "quality-summary.json")["candidates"]
    quality_by = {row["candidate"]: row for row in quality}
    baseline_quality = quality_by.get("A")
    if not baseline_quality or not (baseline_quality["correct"] >= 46 and baseline_quality["complete"] == 48):
        result = {
            "schema": "stage3-analysis-v2",
            "status": "stopped_before_formal_comparison",
            "stop_reason": "A failed the preregistered 46/48 quality gate",
            "quality": quality,
            "measured_facts": [
                "The compatibility matrix completed for all candidates that were attempted.",
                "The fixed A quality set did not reach the required 46/48 gate.",
                "No formal throughput comparison or configuration recommendation was authorized after this gate failed.",
            ],
            "inferences": [
                "The repeated failures in the same question categories may indicate retrieval difficulty or question-design sensitivity, but this is not resolved by the retained evidence.",
            ],
            "unresolved_questions": [
                "Whether the failed questions are model-quality failures or an ambiguity in the fixed question wording.",
                "No claim is made about the relative throughput of the six configurations.",
            ],
        }
        write_json(OUT / "analysis.json", result)
        write_json(OUT / "selection.json", {"selected_candidate": None, "passes_all_gates": False,
                                             "stop_reason": result["stop_reason"]})
        print(json.dumps({"status": result["status"], "stop_reason": result["stop_reason"]}, ensure_ascii=False, indent=2))
        return
    formal = read(OUT / "formal-summary.json")
    compat_by = {row["candidate"]: row for row in compatibility}
    labels = formal["eligible"]
    rows: list[dict[str, Any]] = []
    for label in labels:
        rows.append(round_metrics(formal["rounds"], label))
    by = {row["candidate"]: row for row in rows}
    baseline = by.get("A")
    if baseline is None:
        raise SystemExit("A has no formal result")

    decisions = []
    for row in rows:
        label = row["candidate"]
        quality_row = quality_by[label]
        tpot_changes = [relative_change(row["shared_followup_p95_tpot_seconds"], baseline["shared_followup_p95_tpot_seconds"]),
                        relative_change(row["independent_p95_tpot_seconds"], baseline["independent_p95_tpot_seconds"])]
        e2e_changes = [relative_change(row["shared_followup_p95_e2e_seconds"], baseline["shared_followup_p95_e2e_seconds"]),
                       relative_change(row["independent_p95_e2e_seconds"], baseline["independent_p95_e2e_seconds"])]
        ttft_change = relative_change(row["independent_p95_ttft_seconds"], baseline["independent_p95_ttft_seconds"])
        stable = row["round_count"] == 3 and row["shared_throughput_cv_percent"] is not None and row["shared_throughput_cv_percent"] <= 5
        gates = {
            "quality": quality_row["correct"] >= max(46, baseline_quality["correct"] - 1) and quality_row["complete"] == 48,
            "tpot_regression": all(change is not None and change <= 10 for change in tpot_changes),
            "e2e_regression": all(change is not None and change <= 15 for change in e2e_changes),
            "independent_ttft_regression": ttft_change is not None and ttft_change <= 15,
            "formal_stability": stable,
        }
        decisions.append({**row, "quality": quality_row, "relative_changes": {"p95_tpot": tpot_changes, "p95_e2e": e2e_changes, "independent_p95_ttft": ttft_change}, "gates": gates,
                          "passes_all_gates": all(gates.values()), "cache_capacity_tokens": cache_capacity(compat_by[label])})

    passing = [row for row in decisions if row["passes_all_gates"]]
    # Sorting encodes the preregistered tie-break order. Throughput within 5%
    # uses lower shared-followup TTFT, then cache capacity, then simpler FA/BF16.
    chosen = None
    if passing:
        highest = max(row["shared_throughput_mean"] for row in passing)
        contenders = [row for row in passing if row["shared_throughput_mean"] >= highest * 0.95]
        chosen = sorted(contenders, key=lambda row: (
            row["shared_followup_p95_ttft_seconds"] if row["shared_followup_p95_ttft_seconds"] is not None else float("inf"),
            -(row["cache_capacity_tokens"] or 0),
            0 if compat_by[row["candidate"]]["expected"]["attention_backend"] == "FLASH_ATTN" else 1,
            0 if compat_by[row["candidate"]]["expected"]["kv_cache_dtype"] == "bfloat16" else 1,
        ))[0]

    result = {
        "schema": "stage3-analysis-v2",
        "source_hashes": {"compatibility": sha256_bytes((OUT / "compatibility.json").read_bytes()),
                          "quality": sha256_bytes((OUT / "quality-summary.json").read_bytes()),
                          "formal": sha256_bytes((OUT / "formal-summary.json").read_bytes())},
        "baseline": baseline,
        "decisions": decisions,
        "passing_candidates": [row["candidate"] for row in passing],
        "selected_candidate": None if chosen is None else chosen["candidate"],
        "clear_service_benefit": None if chosen is None else (
            chosen["shared_throughput_mean"] >= baseline["shared_throughput_mean"] * 1.10 or
            chosen["shared_followup_p95_ttft_seconds"] <= baseline["shared_followup_p95_ttft_seconds"] * 0.80
        ),
        "measured_facts": [
            "All values in this file are derived from retained request summaries and compatibility snapshots.",
            "The fixed 48-question set is a bounded quality check, not a general model-quality guarantee.",
        ],
        "inferences": [
            "A difference between backend/dtype/prefix combinations is a configuration effect of existing vLLM mechanisms.",
            "A cache-capacity difference is evidence from reported token blocks, not proof of end-to-end memory savings by itself.",
        ],
        "unresolved_questions": [
            "These WSL measurements do not establish native-Linux or production-arrival behavior.",
            "The fixed Chinese materials do not generalize to all long-document QA quality.",
        ],
    }
    write_json(OUT / "analysis.json", result)
    if chosen is not None:
        write_json(OUT / "selection.json", {
            "selected_candidate": chosen["candidate"],
            "target_input_tokens": formal["target_input_tokens"],
            "client_concurrency": formal["client_concurrency"],
            "selection_reason": "highest shared-load throughput among passing candidates with preregistered tie-breaks",
            "passes_all_gates": True,
            "clear_service_benefit": result["clear_service_benefit"],
        })
    else:
        write_json(OUT / "selection.json", {"selected_candidate": None, "passes_all_gates": False,
                                             "stop_reason": "no candidate passed all preregistered quality, latency, and stability gates"})
    print(json.dumps({"passing": result["passing_candidates"], "selected": result["selected_candidate"], "clear_service_benefit": result["clear_service_benefit"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
