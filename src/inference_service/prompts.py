def document_prompt(document: str, question: str, *, independent_title: bool = False) -> str:
    """Keep document bytes before the changing question for prefix reuse."""
    prompt = ("请只根据以下文档回答末尾的问题。没有依据时说明文档未提供。"
              "简洁回答，保留所需事实，不要展示思考过程。\n\n文档：\n"
              + document + "\n\n问题：" + question)
    return document.split("。", 1)[0] + "\n" + prompt if independent_title else prompt
