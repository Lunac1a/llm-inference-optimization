"""Frozen question-only routing for the bounded integer-answer API."""
from importlib.resources import files
import json

POLICIES = json.loads(files("inference_service").joinpath("policies.json").read_text())


def task_type(question: str) -> str:
    q = question.lower()
    if "record" in q and "value" in q:
        return "lookup"
    if "subsets" in q and "exactly" in q:
        return "counting"
    if "remaining" in q or "remainder" in q:
        return "sequential"
    if "calculate" in q:
        return "arithmetic"
    return "unknown"


def select_budget(question: str, profile: str = "economy") -> tuple[str, int]:
    kind = task_type(question)
    return kind, POLICIES[profile][kind]


def solve_request(question: str, profile: str, model: str, stream: bool) -> tuple[dict, dict]:
    kind, budget = select_budget(question, profile)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": question +
                      "\nReturn only the final integer as your answer, without explanation."}],
        "temperature": 0, "seed": 42, "max_tokens": budget + 192,
        "chat_template_kwargs": {"enable_thinking": budget > 0},
        "thinking_token_budget": budget,
        "structured_outputs": {"regex": "[ \\n]{0,4}-?[0-9]{1,8}[ \\n]{0,2}"},
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    return body, {"profile": profile, "task_type": kind, "thinking_tokens": budget}
