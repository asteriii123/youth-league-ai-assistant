"""
Prompt模板管理模块。

负责：

集中管理AI系统提示词。


例如：

- RAG回答提示词
- 会议总结提示词
- Agent角色定义


避免Prompt散落在代码中。
"""
def rag_system_prompt(role: str, parents: list[dict]) -> str:
    role_prompt = (
        "协助团支书起草通知、整理工作计划、准备会议和解答团务常见问题。"
        if role == "secretary"
        else "帮助学生理解入团入党材料、团员义务和团务常见问题。"
    )
    rules = (
        "回答使用简洁、友善的中文。对政策、流程、时间和材料要求等确定性问题，只能依据下方本班资料回答，"
        "不得虚构政策、文件、日期或来源；每项资料结论都用[资料1]格式标注。资料不足时明确说‘知识库依据不足’，"
        "不要伪造引用。可以提供通用建议，但必须明确标注为通用建议，并与资料结论分开。"
    )
    if not parents:
        return f"你是高校团务AI助手，{role_prompt}{rules}当前没有检索到可用资料。"
    sources = []
    for item in parents:
        location = f"第{item['page']}页" if item.get("page") else "页码未知"
        heading = item.get("section_path") or item.get("heading") or "未命名章节"
        sources.append(
            f"[{item['source_label']}] 文件：{item['filename']}；章节：{heading}；{location}\n{item['content']}"
        )
    return f"你是高校团务AI助手，{role_prompt}{rules}\n\n本班知识资料：\n" + "\n\n".join(sources)
