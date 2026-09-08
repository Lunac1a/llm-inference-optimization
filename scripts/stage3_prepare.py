"""Create deterministic long-document and quality materials with the pinned tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from stage3_lib import OUT, OwnedVllm, candidate_config, sha256_bytes, write_json


DOC_SPECS = [
    {
        "id": "doc-a",
        "title": "云桥城市数据项目甲卷",
        "code": "云桥甲-17",
        "date": "2026年5月12日",
        "owner": "林若安",
        "budget": "480万元",
        "region": "墨尔本北区",
        "relation": "先校验数据层，再执行模型评估，最后提交服务报告",
        "risk": "供应商接口延迟",
        "distractor_budget": "820万元",
    },
    {
        "id": "doc-b",
        "title": "澄海物流预测项目乙卷",
        "code": "澄海乙-29",
        "date": "2026年6月3日",
        "owner": "周明川",
        "budget": "315万元",
        "region": "悉尼西区",
        "relation": "先清理路线数据，再训练预测模型，最后进行人工抽检",
        "risk": "节假日样本不足",
        "distractor_budget": "560万元",
    },
    {
        "id": "doc-c",
        "title": "星环客服检索项目丙卷",
        "code": "星环丙-08",
        "date": "2026年4月21日",
        "owner": "顾清禾",
        "budget": "260万元",
        "region": "布里斯班南区",
        "relation": "先整理知识库，再进行检索评测，最后接入客服试点",
        "risk": "历史工单缺少标签",
        "distractor_budget": "410万元",
    },
    {
        "id": "doc-d",
        "title": "远岚制造质检项目丁卷",
        "code": "远岚丁-41",
        "date": "2026年7月8日",
        "owner": "沈砚舟",
        "budget": "730万元",
        "region": "珀斯东区",
        "relation": "先标定摄像头，再验证缺陷样本，最后部署质检接口",
        "risk": "夜班图像亮度波动",
        "distractor_budget": "990万元",
    },
]


def import_tokenizer(model_path: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Run this script with the pinned inference venv") from exc
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)


def ids(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def exact_length(tokenizer, front: str, middle: str, back: str, filler: str, target: int) -> tuple[str, int]:
    front_ids = ids(tokenizer, front)
    middle_ids = ids(tokenizer, middle)
    back_ids = ids(tokenizer, back)
    filler_ids = ids(tokenizer, filler)
    base = len(front_ids) + len(middle_ids) + len(back_ids)
    if base >= target:
        raise ValueError(f"base facts exceed target {target}: {base}")
    remaining = target - base
    first = remaining // 2
    second = remaining - first

    def repeated(count: int) -> list[int]:
        return (filler_ids * (count // len(filler_ids) + 1))[:count]

    token_ids = front_ids + repeated(first) + middle_ids + repeated(second) + back_ids
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    actual_ids = ids(tokenizer, text)
    # Decode can merge whitespace around a boundary. Preserve the measured count
    # and add a short deterministic suffix until the request reaches the target.
    while len(actual_ids) < target:
        actual_ids += filler_ids[: min(len(filler_ids), target - len(actual_ids))]
        text = tokenizer.decode(actual_ids, skip_special_tokens=True)
        actual_ids = ids(tokenizer, text)
    if len(actual_ids) > target:
        text = tokenizer.decode(actual_ids[:target], skip_special_tokens=True)
        actual_ids = ids(tokenizer, text)
    return text, len(actual_ids)


def make_document(tokenizer, spec: dict, target: int) -> dict:
    steps = spec['relation'].split('，')
    front = (
        f"{spec['title']}。这是性能与质量测试材料，文档编号为{spec['code']}。"
        f"发布日期是{spec['date']}，所在区域是{spec['region']}。"
        f"正式流程第一步：{steps[0]}。后续步骤分别见中部与末尾记录。\n"
        "以下内容是项目说明，不应把相邻章节中的示例数字当成正式事实。\n"
    )
    middle = (
        f"负责人记录：项目负责人是{spec['owner']}。正式流程第二步：{steps[1]}。\n"
        f"风险记录：当前已知风险是{spec['risk']}；风险记录不代表项目已经失败。\n"
        f"干扰信息：演示预算曾写作{spec['distractor_budget']}，它是培训示例，不是本项目预算。\n"
    )
    back = (
        f"最终确认：文档明确记载的正式预算是{spec['budget']}。"
        f"正式流程第三步：{steps[2]}。文档没有给出预计日活用户数。\n"
    )
    text, actual = exact_length(
        tokenizer,
        front,
        middle,
        back,
        "本段为固定种子生成的背景说明，用于保持长上下文测试长度。\n",
        target,
    )
    return {
        "id": spec["id"],
        "title": spec["title"],
        "target_tokens": target,
        "actual_document_tokens": actual,
        "sha256": sha256_bytes(text.encode("utf-8")),
        "text": text,
        "facts": {key: spec[key] for key in ("code", "date", "owner", "budget", "region", "relation", "risk")},
    }


def quality_questions(doc: dict, spec: dict) -> list[dict]:
    prefix = "只根据下面资料回答。如果资料没有明确给出，请回答‘资料未提供’，不要猜测。\n\n"
    text = doc["text"]
    common = {"doc_id": doc["id"], "document_sha256": doc["sha256"], "document": text}
    return [
        {**common, "id": f"{doc['id']}-direct-1", "category": "direct", "location": "front", "question": "项目编号是什么？", "answer_groups": [[spec["code"]]]},
        {**common, "id": f"{doc['id']}-direct-2", "category": "direct", "location": "middle", "question": "项目负责人是谁？", "answer_groups": [[spec["owner"]]]},
        {**common, "id": f"{doc['id']}-direct-3", "category": "direct", "location": "back", "question": "文档明确记载的预算是多少？", "answer_groups": [[spec["budget"]]]},
        {**common, "id": f"{doc['id']}-cross-1", "category": "cross_paragraph", "location": "front_middle_back", "question": "请给出该项目的正式执行顺序。", "answer_groups": [[spec["relation"]]]},
        {**common, "id": f"{doc['id']}-cross-2", "category": "cross_paragraph", "location": "front_middle", "question": "将项目编号与负责人一起写出。", "answer_groups": [[spec["code"]], [spec["owner"]]]},
        {**common, "id": f"{doc['id']}-cross-3", "category": "cross_paragraph", "location": "back", "question": "正式执行顺序的最后一步是什么？请原样写出以“最后”开头的步骤。", "answer_groups": [[spec["relation"].split("，")[-1]]]},
        {**common, "id": f"{doc['id']}-distractor-1", "category": "distractor", "location": "middle_back", "question": "本项目的正式预算金额是多少？请只回答金额及单位，忽略培训示例预算。", "answer_groups": [[spec["budget"]]]},
        {**common, "id": f"{doc['id']}-distractor-2", "category": "distractor", "location": "middle", "question": "已知风险是什么？不要把风险描述成预算或流程。", "answer_groups": [[spec["risk"]]]},
        {**common, "id": f"{doc['id']}-distractor-3", "category": "distractor", "location": "front", "question": "项目所在区域是什么？不要把培训示例数字当作区域。", "answer_groups": [[spec["region"]]]},
        {**common, "id": f"{doc['id']}-missing-1", "category": "missing", "location": "global_absence", "question": "文档给出的预计日活用户数是多少？", "answer_groups": [["资料未提供", "未提供", "无法确定", "没有给出"]]},
        {**common, "id": f"{doc['id']}-missing-2", "category": "missing", "location": "global_absence", "question": "文档是否给出了项目的最终利润率？", "answer_groups": [["资料未提供", "未提供", "无法确定", "没有给出"]]},
        {**common, "id": f"{doc['id']}-missing-3", "category": "missing", "location": "global_absence", "question": "文档是否明确给出了参与项目的员工人数？", "answer_groups": [["资料未提供", "未提供", "无法确定", "没有给出"]]},
    ]


def build_materials(model_path: str) -> dict:
    tokenizer = import_tokenizer(model_path)
    documents: dict[str, dict[str, dict]] = {}
    for target in (8192, 16384):
        documents[str(target)] = {}
        for spec in DOC_SPECS:
            documents[str(target)][spec["id"]] = make_document(tokenizer, spec, target)

    short = (
        "这是独立编译预热材料。请回答：固定测试只要求返回‘预热完成’，不要解释。"
    )
    performance: dict[str, list[dict]] = {}
    for target in (8192, 16384):
        performance[str(target)] = []
        for index in range(32):
            doc_id = DOC_SPECS[index % len(DOC_SPECS)]["id"]
            document = documents[str(target)][doc_id]
            performance[str(target)].append({
                "id": f"perf-{target}-{index + 1:02d}",
                "doc_id": doc_id,
                "prompt": f"请阅读以下固定测试文档，并用简洁中文回答问题。\n\n{document['text']}\n\n问题：请概括该文档的项目编号和正式流程。",
                "document_sha256": document["sha256"],
                "target_tokens": target,
                "actual_prompt_document_tokens": document["actual_document_tokens"],
            })

    quality = []
    for spec in DOC_SPECS:
        quality_spec = {**spec, "id": "quality-" + spec["id"],
                        "code": "验收-" + spec["code"], "title": "独立验收材料：" + spec["title"]}
        doc = make_document(tokenizer, quality_spec, 16384)
        quality.extend(quality_questions(doc, quality_spec))

    return {
        "schema": "stage3-materials-v2",
        "seed": 20260908,
        "tokenizer_model": "Qwen/Qwen3-4B",
        "short_prompt": short,
        "documents": documents,
        "performance": performance,
        "quality": quality,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    model_path = args.model_path
    if not model_path:
        model_path = OwnedVllm(candidate_config("A"), OUT / "prepare").resolve_model()
    if (OUT / "materials.json").exists():
        raise SystemExit("Refuse to overwrite existing materials")
    materials = build_materials(model_path)
    write_json(OUT / "materials.json", materials)
    manifest = {
        "schema": materials["schema"],
        "seed": materials["seed"],
        "documents": {
            target: {doc_id: {key: doc[key] for key in ("target_tokens", "actual_document_tokens", "sha256")}
                     for doc_id, doc in docs.items()}
            for target, docs in materials["documents"].items()
        },
        "quality_count": len(materials["quality"]),
        "performance_counts": {target: len(rows) for target, rows in materials["performance"].items()},
    }
    write_json(OUT / "materials-manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
