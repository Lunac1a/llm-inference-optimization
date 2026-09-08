"""Bounded Stage 3 collector for six local vLLM KV-cache combinations."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import itertools
from pathlib import Path
import re
import statistics
import time
import unicodedata
from typing import Any

from stage3_lib import (
    Budget,
    OUT,
    OwnedVllm,
    Telemetry,
    append_jsonl,
    candidate_config,
    chat_request,
    http_json,
    load_config,
    metric_delta,
    metrics_snapshot,
    now_utc,
    percentile,
    reset_prefix_cache,
    sha256_json,
    write_json,
)


MODEL_NAME_PREFIX = "stage3-"


def materials() -> dict[str, Any]:
    path = OUT / "materials.json"
    if not path.exists():
        raise SystemExit("Missing artifacts/stage3/materials.json; run phase prepare first")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "stage3-materials-v2":
        raise ValueError("Refuse to mix archived v1 materials with the corrected collector")
    return data


def base_url(candidate: dict[str, Any]) -> str:
    return f"http://{candidate['host']}:{candidate['port']}"


def model_name(candidate: dict[str, Any]) -> str:
    return f"{MODEL_NAME_PREFIX}{candidate['label'].lower()}"


def prompt_hash(prompt: str) -> str:
    return sha256_json({"prompt": prompt})


def prompt_for_quality(row: dict[str, Any]) -> str:
    return (
        "只根据下面资料回答。如果资料没有明确给出，请回答‘资料未提供’，不要猜测。"
        "只输出答案值，不要复述问题或解释；多项答案用分号分隔，不要展示思考过程。\n\n"
        f"{row['document']}\n\n问题：{row['question']}"
    )


def score_quality_answer(question: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    """Deterministic bounded scorer; truncation and empty answers always fail."""
    answer = response.get("text", "") or ""
    complete = (response.get("status") == 200 and response.get("stream_complete") is True
                and bool(answer.strip()) and response.get("finish_reason") == "stop"
                and not response.get("error"))
    groups = question.get("answer_groups")
    if not groups or any(not group for group in groups):
        raise ValueError("Quality scoring requires the v2 answer_groups schema")
    def normalize(value):
        return ''.join(c for c in unicodedata.normalize('NFKC', value)
                       if not c.isspace() and not unicodedata.category(c).startswith('P'))
    acceptable = {normalize(''.join(order)) for values in itertools.product(*groups)
                  for order in itertools.permutations(values)}
    correct = complete and normalize(answer) in acceptable
    return {"question": question, "response": response, "complete": complete, "correct": correct}


def prompt_for_performance(row: dict[str, Any], independent_prefix: str = "") -> str:
    return independent_prefix + row["prompt"]


def request_one(candidate: dict[str, Any], prompt: str, max_tokens: int, ignore_eos: bool, request_id: str, timeout: float | None = None) -> dict[str, Any]:
    result = chat_request(base_url(candidate), prompt, model_name(candidate), max_tokens, stream=True, ignore_eos=ignore_eos,
                          timeout=candidate.get("request_timeout_seconds", 120) if timeout is None else timeout)
    result.update({"request_id": request_id, "prompt_sha256": prompt_hash(prompt), "prompt_chars": len(prompt), "captured_at": time.time()})
    return result


def run_batch(
    candidate: dict[str, Any],
    phase_dir: Path,
    batch_name: str,
    prompts: list[tuple[str, str]],
    concurrency: int,
    max_tokens: int,
    ignore_eos: bool,
    warmups: list[tuple[str, str]] | None = None,
    budget: Budget | None = None,
) -> dict[str, Any]:
    """Run a bounded batch and retain one raw response object per request."""
    budget = budget or Budget()
    batch_timeout = candidate.get("batch_timeout_seconds", 600)
    budget.ensure(batch_timeout, candidate.get("cleanup_reserve_seconds", 300))
    phase_dir.mkdir(parents=True, exist_ok=True)
    batch_dir = phase_dir / batch_name
    batch_dir.mkdir(parents=True, exist_ok=False)
    url = base_url(candidate)
    deadline = time.perf_counter() + batch_timeout
    started_epoch = time.time()

    def bounded_request(request_id, prompt):
        remaining = min(candidate.get("request_timeout_seconds", 120), deadline - time.perf_counter())
        return request_one(candidate, prompt, max_tokens, ignore_eos, request_id, timeout=remaining)

    warmup_rows: list[dict[str, Any]] = []
    for request_id, prompt in warmups or []:
        warmup_rows.append(bounded_request(request_id, prompt))
        if not warmup_rows[-1].get("stream_complete") or warmup_rows[-1].get("error") or (ignore_eos and warmup_rows[-1].get("output_tokens") != max_tokens):
            write_json(batch_dir / "warmups.json", warmup_rows)
            raise RuntimeError("Warmup failed; stop branch")
    write_json(batch_dir / "warmups.json", warmup_rows)

    before = metrics_snapshot(url)
    telemetry = Telemetry(url, batch_dir / "telemetry.jsonl")
    telemetry.start()
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="stage3-request") as pool:
        futures = {pool.submit(bounded_request, request_id, prompt): request_id
                   for request_id, prompt in prompts}
        try:
            for future in as_completed(futures, timeout=max(0, deadline - time.perf_counter())):
                try:
                    row = future.result()
                except Exception as exc:
                    row = {"request_id": futures[future], "error": f"{type(exc).__name__}: {exc}", "status": None, "e2e_seconds": None}
                rows.append(row)
                if row.get("error") or row.get("status") != 200:
                    errors.append(str(row.get("error") or row.get("status")))
        except TimeoutError:
            errors.append("batch_deadline_exceeded")
            for future in futures:
                future.cancel()
    # Running HTTP subprocesses share the same deadline and are killed/reaped;
    # exiting the thread pool cannot wait for an unbounded trickling connection.
    observed = {row["request_id"] for row in rows}
    for request_id, _ in prompts:
        if request_id not in observed:
            rows.append({"request_id": request_id, "status": None, "error": "batch_deadline_exceeded", "stream_complete": False})
    wall_seconds = time.perf_counter() - started
    telemetry.stop()
    after = metrics_snapshot(url)
    append_jsonl(batch_dir / "requests.jsonl", {"batch_started_at": started_epoch, "rows": rows})
    # Also keep the request rows as a normal JSON object for easy audit.
    write_json(batch_dir / "requests.json", rows)

    successes = [row for row in rows if row.get("status") == 200 and not row.get("error") and row.get("stream_complete")]
    ttft = [row["ttft_seconds"] for row in successes if row.get("ttft_seconds") is not None]
    e2e = [row["e2e_seconds"] for row in successes if row.get("e2e_seconds") is not None]
    tpot = []
    output_tokens = []
    for row in successes:
        count = row.get("output_tokens")
        if isinstance(count, (int, float)) and count > 0:
            output_tokens.append(float(count))
            if row.get("ttft_seconds") is not None and count > 1:
                tpot.append((row["e2e_seconds"] - row["ttft_seconds"]) / (count - 1))
    invalid_output_lengths = [
        row for row in successes
        if ignore_eos and row.get("output_tokens") != max_tokens
    ]
    summary = {
        "batch": batch_name,
        "candidate": candidate["label"],
        "candidate_config": candidate,
        "request_count": len(prompts),
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "ignore_eos": ignore_eos,
        "warmup_count": len(warmup_rows),
        "started_at": now_utc(),
        "wall_seconds": wall_seconds,
        "success_count": len(successes),
        "failure_count": len(prompts) - len(successes) + len(invalid_output_lengths),
        "invalid_output_length_count": len(invalid_output_lengths),
        "invalid_output_lengths": [row.get("output_tokens") for row in invalid_output_lengths],
        "errors": errors,
        "output_tokens_sum": sum(output_tokens),
        "output_throughput_tok_s": sum(output_tokens) / wall_seconds if wall_seconds > 0 else None,
        "ttft_seconds": {"p50": percentile(ttft, 50), "p95": percentile(ttft, 95)},
        "tpot_seconds": {"p50": percentile(tpot, 50), "p95": percentile(tpot, 95)},
        "e2e_seconds": {"p50": percentile(e2e, 50), "p95": percentile(e2e, 95)},
        "metrics_before": before,
        "metrics_after": after,
        "metric_deltas": metric_delta(before, after),
        "cache_reset_not_used": True,
    }
    write_json(batch_dir / "summary.json", summary)
    if summary["failure_count"] or errors:
        raise RuntimeError(f"Batch failed; evidence retained at {batch_dir}")
    return summary


def compile_warmups(candidate: dict[str, Any], materials_data: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for index in range(2):
        rows.append(request_one(candidate, materials_data["short_prompt"], 16, False, f"compile-warmup-{index + 1}"))
    write_json(run_dir / "compile-warmups.json", rows)
    if any(not row.get('stream_complete') or row.get('error') for row in rows):
        raise RuntimeError("Compilation warmup failed; stop branch")
    return rows


def start_candidate(candidate: dict[str, Any], label: str, budget: Budget) -> OwnedVllm:
    budget.ensure()
    run_dir = OUT / "runs" / label
    run_dir.mkdir(parents=True, exist_ok=True)
    server = OwnedVllm(candidate, run_dir)
    budget.record("server_start", candidate=candidate["label"], run=label)
    try:
        server.start()
    except BaseException:
        stop_candidate(server, budget, label)
        raise
    return server


def stop_candidate(server: OwnedVllm, budget: Budget, label: str) -> None:
    result = server.stop()
    budget.record("server_stop", candidate=server.candidate["label"], run=label, **result)


def cold_cache(server: OwnedVllm, budget: Budget, label: str) -> dict[str, Any]:
    """Failed/unsupported reset must produce a new healthy owned process."""
    reset = reset_prefix_cache(server.base_url)
    if reset["ok"]:
        return {**reset, "method": "reset"}
    stop_candidate(server, budget, label)
    budget.ensure()
    server.run_dir = server.run_dir.parent / f"{label}-cold-{time.time_ns()}"
    budget.record("server_start", candidate=server.candidate["label"], run=server.run_dir.name)
    try:
        startup = server.start()
    except BaseException:
        stop_candidate(server, budget, label)
        raise
    return {"ok": True, "method": "restart", "reset_attempt": reset, "startup": startup}


def phase_prepare() -> None:
    from stage3_prepare import build_materials

    if (OUT / "materials.json").exists():
        raise RuntimeError("Refuse to overwrite existing materials")
    candidate = candidate_config("A")
    model_path = OwnedVllm(candidate, OUT / "prepare").resolve_model()
    materials_data = build_materials(model_path)
    write_json(OUT / "materials.json", materials_data)
    write_json(OUT / "materials-manifest.json", {
        "schema": materials_data["schema"],
        "seed": materials_data["seed"],
        "documents": {
            target: {doc_id: {key: doc[key] for key in ("target_tokens", "actual_document_tokens", "sha256")}
                     for doc_id, doc in docs.items()}
            for target, docs in materials_data["documents"].items()
        },
        "quality_count": len(materials_data["quality"]),
        "performance_counts": {key: len(value) for key, value in materials_data["performance"].items()},
    })
    print("Prepared Stage 3 materials")


def phase_compatibility(budget: Budget) -> None:
    data = materials()
    rows = []
    for label in load_config()["candidates"]:
        candidate = candidate_config(label)
        run = f"compatibility-{label}-{now_utc()}"
        server = None
        row: dict[str, Any] = {"candidate": label, "expected": candidate, "run": run, "started_at": now_utc()}
        try:
            server = start_candidate(candidate, run, budget)
            warmups = compile_warmups(candidate, data, server.run_dir)
            before = metrics_snapshot(server.base_url)
            request = request_one(candidate, data["short_prompt"], 16, False, f"compat-{label}")
            after = metrics_snapshot(server.base_url)
            log = (server.run_dir / "server.log").read_text(errors="replace")
            backend_match = re.findall(r"(?:FLASH_ATTN|TRITON_ATTN|FlashAttention|TritonAttention)", log, flags=re.I)
            row.update({"status": "compatible" if request.get("status") == 200 and not request.get("error") else "request_failed",
                        "request": request, "warmups": warmups, "metrics_before": before, "metrics_after": after,
                        "cache_config": after.get("cache_config"), "backend_evidence": sorted(set(backend_match)),
                        "log_tail": log[-20000:]})
        except Exception as exc:
            row.update({"status": "incompatible", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            if server is not None:
                stop_candidate(server, budget, run)
        write_json(OUT / "compatibility" / f"{label}.json", row)
        rows.append(row)
    write_json(OUT / "compatibility.json", {"captured_at": now_utc(), "candidates": rows})


def phase_screen(budget: Budget) -> None:
    data = materials()
    candidate = candidate_config("A")
    server = start_candidate(candidate, f"screen-A-{now_utc()}", budget)
    rows = []
    try:
        for target in (8192, 16384):
            perf = data["performance"][str(target)]
            for concurrency in (4, 8):
                warmups = [(f"screen-warmup-{target}-{concurrency}-{i}", perf[i]["prompt"]) for i in range(4)]
                prompts = [(perf[i]["id"], perf[i]["prompt"]) for i in range(8)]
                run = run_batch(candidate, OUT / "screen", f"A-{target}-c{concurrency}", prompts, concurrency, 128, True, warmups)
                run.update({"input_target_tokens": target, "screening_concurrency": concurrency})
                rows.append(run)
                if run["failure_count"]:
                    # The plan permits continuing to the other predeclared point,
                    # but no unplanned load escalation is attempted.
                    continue
    finally:
        stop_candidate(server, budget, "screen-A")
    passing = [row for row in rows if row["failure_count"] == 0 and (row["e2e_seconds"]["p95"] or 999) <= 30]
    selected = None
    if passing:
        max_length = max(row["input_target_tokens"] for row in passing)
        at_length = [row for row in passing if row["input_target_tokens"] == max_length]
        selected = max(at_length, key=lambda row: row["screening_concurrency"])
    write_json(OUT / "screening.json", {"rows": rows, "passing_rows": len(passing), "selection": selected,
                                         "stop_reason": None if selected else "no point met zero-errors and p95-E2E<=30s"})
    if selected is None:
        raise RuntimeError("Stage 3 stopped before formal comparison: no screening point passed")


def compatible_labels() -> list[str]:
    path = OUT / "compatibility.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["candidates"]
    return [row["candidate"] for row in rows if row.get("status") == "compatible"]


def phase_quality(budget: Budget) -> None:
    data = materials()
    labels = compatible_labels()
    all_summaries = []
    for label in labels:
        candidate = candidate_config(label)
        server = start_candidate(candidate, f"quality-{label}-{now_utc()}", budget)
        run_dir = OUT / "quality" / label
        rows = []
        try:
            reset = cold_cache(server, budget, f"quality-{label}")
            write_json(run_dir / "cache-reset.json", reset)
            prompts = [(row["id"], prompt_for_quality(row)) for row in data["quality"]]
            result = run_batch(candidate, run_dir, "all", prompts, 4, 256, False)
            raw_rows = json.loads((run_dir / "all" / "requests.json").read_text(encoding="utf-8"))
            by_id = {row["request_id"]: row for row in raw_rows}
            for question in data["quality"]:
                response = by_id.get(question["id"], {})
                scored = score_quality_answer(question, response)
                rows.append(scored)
                append_jsonl(run_dir / "scored.jsonl", scored)
            correct_count = sum(row["correct"] for row in rows)
            complete_count = sum(row["complete"] for row in rows)
            summary = {"candidate": label, "question_count": len(rows), "correct": correct_count,
                       "complete": complete_count, "accuracy": correct_count / len(rows) if rows else 0,
                       "minimum_pass": correct_count >= 46 and complete_count == 48,
                       "batch": result, "cache_reset": reset}
            write_json(run_dir / "summary.json", summary)
            all_summaries.append(summary)
        finally:
            stop_candidate(server, budget, f"quality-{label}")
    write_json(OUT / "quality-summary.json", {"candidates": all_summaries,
                                               "A_passed": next((row["minimum_pass"] for row in all_summaries if row["candidate"] == "A"), False)})
    if not any(row["candidate"] == "A" and row["minimum_pass"] for row in all_summaries):
        raise RuntimeError("Stage 3 stopped: A did not reach 46/48 complete correct answers")


def formal_prompts(data: dict[str, Any], target: int, shared: bool) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    docs = data["documents"][str(target)]
    doc_ids = sorted(docs)
    first = []
    for doc_id in doc_ids:
        doc = docs[doc_id]
        first.append((f"first-{doc_id}", document_prompt(doc['text'], "项目编号是什么？")))
    followups = []
    for index in range(16):
        doc_id = doc_ids[index % len(doc_ids)]
        doc = docs[doc_id]
        question = f"第{index + 1}个后续问题：请回答该文档中记录的项目负责人和正式流程。"
        prompt = document_prompt(doc['text'], question)
        followups.append((f"followup-{index + 1:02d}-{doc_id}", prompt))
    if not shared:
        first = []
        followups = [(request_id, f"独立请求标识：{request_id}。" + prompt) for request_id, prompt in followups]
    return first, followups


def document_prompt(document: str, question: str) -> str:
    return f"请阅读以下固定文档并简洁回答问题。\n\n{document}\n\n问题：{question}"


def phase_formal(budget: Budget) -> None:
    data = materials()
    screen = json.loads((OUT / "screening.json").read_text(encoding="utf-8"))
    selected = screen["selection"]
    target = selected["input_target_tokens"]
    concurrency = selected["screening_concurrency"]
    quality = json.loads((OUT / "quality-summary.json").read_text(encoding="utf-8"))["candidates"]
    baseline_quality = next(row for row in quality if row["candidate"] == "A")
    quality_pass = {row["candidate"] for row in quality if row["minimum_pass"]
                    and row["correct"] >= baseline_quality["correct"] - 1}
    compatible = compatible_labels()
    eligible = [label for label in load_config()["candidates"] if label in compatible and label in quality_pass]
    if "A" not in eligible:
        raise RuntimeError("Stage 3 stopped: A is not eligible for formal comparison")
    formal_rows = []
    for round_index in range(3):
        full_order = list(load_config()["candidates"])
        rotation = (2 * round_index) % len(full_order)
        order = full_order[rotation:] + full_order[:rotation]
        for label in [item for item in order if item in eligible]:
            candidate = candidate_config(label)
            run = f"formal-r{round_index + 1}-{label}-{now_utc()}"
            server = start_candidate(candidate, run, budget)
            record: dict[str, Any] = {"round": round_index + 1, "candidate": label, "order": order,
                                      "target_input_tokens": target, "client_concurrency": concurrency}
            try:
                compile_warmups(candidate, data, server.run_dir)
                first, followups = formal_prompts(data, target, True)
                shared_reset = cold_cache(server, budget, f"{run}-shared")
                shared_first = run_batch(candidate, OUT / "formal" / run / "shared", "first-visits",
                                         first, min(4, concurrency), 128, True,
                                         [(f"shared-warmup-{i}", data["short_prompt"]) for i in range(2)])
                shared_follow = run_batch(candidate, OUT / "formal" / run / "shared", "followups",
                                          followups, concurrency, 128, True)
                shared_total_output = shared_first["output_tokens_sum"] + shared_follow["output_tokens_sum"]
                shared_total_wall = shared_first["wall_seconds"] + shared_follow["wall_seconds"]
                shared = {"cache_reset": shared_reset, "first_visits": shared_first, "followups": shared_follow,
                          "including_first_throughput_tok_s": shared_total_output / shared_total_wall if shared_total_wall else None}

                independent_reset = cold_cache(server, budget, f"{run}-independent")
                _, independent = formal_prompts(data, target, False)
                independent_result = run_batch(candidate, OUT / "formal" / run / "independent", "requests",
                                               independent, concurrency, 128, True,
                                               [(f"independent-warmup-{i}", data["short_prompt"]) for i in range(2)])
                record.update({"shared": shared, "independent": {"cache_reset": independent_reset, "requests": independent_result}})
                write_json(OUT / "formal" / f"r{round_index + 1}-{label}.json", record)
                formal_rows.append(record)
            finally:
                stop_candidate(server, budget, run)
    write_json(OUT / "formal-summary.json", {"target_input_tokens": target, "client_concurrency": concurrency,
                                              "eligible": eligible, "rounds": formal_rows})


def phase_acceptance(budget: Budget) -> None:
    selection = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    label = selection["selected_candidate"]
    candidate = candidate_config(label)
    data = materials()
    run = f"acceptance-{label}-{now_utc()}"
    server = start_candidate(candidate, run, budget)
    result: dict[str, Any] = {"candidate": label, "run": run, "checks": []}
    try:
        doc = data["documents"][str(selection["target_input_tokens"])]["doc-a"]
        normal = chat_request(server.base_url, f"请回答文档中的项目编号。\n\n{doc['text']}", model_name(candidate), 128, stream=False)
        result["checks"].append({"name": "normal_response", "passed": normal.get("status") == 200 and bool(normal.get("text")), "response": normal})
        streams = []
        for index, question in enumerate(("项目负责人是谁？", "正式流程是什么？"), 1):
            streams.append(request_one(candidate, f"请根据以下文档回答问题：\n\n{doc['text']}\n\n问题：{question}", 128, False, f"stream-{index}"))
        result["checks"].append({"name": "streaming_two_questions", "passed": all(row.get("status") == 200 and row.get("stream_complete") for row in streams), "responses": streams})
        other = data["documents"][str(selection["target_input_tokens"])]["doc-b"]
        switched = request_one(candidate, f"切换到另一份文档。请回答编号。\n\n{other['text']}", 128, False, "document-switch")
        result["checks"].append({"name": "document_switch", "passed": switched.get("status") == 200 and "澄海乙-29" in switched.get("text", ""), "response": switched})
    finally:
        stop_candidate(server, budget, run)

    cold_run = f"acceptance-cold-{label}-{now_utc()}"
    cold = start_candidate(candidate, cold_run, budget)
    try:
        doc = data["documents"][str(selection["target_input_tokens"])]["doc-a"]
        cold_response = request_one(candidate, f"冷缓存请求：请回答项目编号。\n\n{doc['text']}", 128, False, "cold-cache")
        result["checks"].append({"name": "restart_cold_cache", "passed": cold_response.get("status") == 200 and bool(cold_response.get("text")), "response": cold_response})
        too_long = (doc["text"] + "\n") * 3
        status, body, elapsed = http_json(f"{cold.base_url}/v1/chat/completions", {
            "model": model_name(candidate), "messages": [{"role": "user", "content": too_long}],
            "temperature": 0, "max_tokens": 8, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }, timeout=120)
        result["checks"].append({"name": "oversized_input_rejected", "passed": status in (400, 413), "status": status, "body": body, "elapsed_seconds": elapsed})
    finally:
        stop_candidate(cold, budget, cold_run)
    status, _, _ = http_json(f"{base_url(candidate)}/health", timeout=3)
    result["port_clean_after_stop"] = status != 200
    result["passed"] = all(check["passed"] for check in result["checks"]) and result["port_clean_after_stop"]
    write_json(OUT / "acceptance.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "compatibility", "screen", "quality", "formal", "acceptance"])
    args = parser.parse_args()
    phase_outputs = {'prepare': 'materials.json', 'compatibility': 'compatibility',
                     'screen': 'screen', 'quality': 'quality', 'formal': 'formal',
                     'acceptance': 'acceptance.json'}
    if (OUT / phase_outputs[args.phase]).exists():
        raise SystemExit("This phase already has evidence; refusing a repeat/overwrite")
    if args.phase == "prepare":
        phase_prepare()
        return
    budget = Budget()
    if args.phase == "compatibility":
        phase_compatibility(budget)
    elif args.phase == "screen":
        phase_screen(budget)
    elif args.phase == "quality":
        phase_quality(budget)
    elif args.phase == "formal":
        phase_formal(budget)
    else:
        phase_acceptance(budget)
    budget.record("phase_complete", phase=args.phase)


if __name__ == "__main__":
    main()
