"""
AI能力接口模块。

负责：

1. 用户AI聊天
2. RAG增强问答
3. DeepSeek大模型调用
4. 学生问题咨询
5. 会议文字总结


核心流程：

用户问题
    ↓
是否需要RAG检索
    ↓
知识库搜索
    ↓
DeepSeek生成答案
    ↓
流式返回前端


主要依赖：

- DeepSeek LLM
- RAG Retrieval
- Prompt模板

对应功能：

团支书AI助手聊天入口。
"""
import json
import os
import re
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, ValidationError
from app.core.config import settings
from app.core.security import get_current_user, require_secretary
from app.llm.deepseek import DeepSeekClient, DeepSeekError, get_deepseek_client
from app.llm.prompts import rag_system_prompt
from app.models.entities import User
from app.rag.retrieval import RetrievalError, retrieve_with_rerank


router = APIRouter(prefix="/api/ai", tags=["AI"])

QA_DISCLAIMER = "当前版本尚未接入本地知识库，回答仅供测试参考，请以学校和学院正式文件为准。"


class StudentQuestionRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)


class StudentAnswerResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    has_reliable_source: bool = False
    disclaimer: str = QA_DISCLAIMER


class MeetingSummaryRequest(BaseModel):
    meeting_type: Literal["主题团日", "团课", "支部会议", "其他"]
    transcript: str = Field(min_length=20, max_length=100000)


class ActionItem(BaseModel):
    task: str
    owner: str = "未指定"
    deadline: str = "未指定"


class MeetingSummaryResponse(BaseModel):
    title: str
    meeting_type: str
    summary: str
    key_points: list[str]
    decisions: list[str]
    action_items: list[ActionItem]
    requires_manual_review: bool = True
    redacted_sensitive_data: bool


def redact_sensitive_text(text: str) -> tuple[str, bool]:
    patterns = [
        (r"(?<!\d)1[3-9]\d{9}(?!\d)", "[手机号已隐藏]"),
        (r"(?<!\d)\d{17}[\dXx](?!\d)", "[身份证号已隐藏]"),
        (r"(学号[：:\s]*)([A-Za-z0-9_-]{6,20})", r"\1[已隐藏]"),
        (r"(姓名[：:\s]*)([\u4e00-\u9fff]{2,4})", r"\1[已隐藏]"),
    ]
    redacted = text
    for pattern, replacement in patterns:
        redacted = re.sub(pattern, replacement, redacted)
    return redacted, redacted != text


def ai_http_error(exc: DeepSeekError) -> HTTPException:
    status_code = 503 if "API_KEY" in str(exc) else 502
    return HTTPException(status_code=status_code, detail=str(exc))


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=12)


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    user: User = Depends(get_current_user),
    client: DeepSeekClient = Depends(get_deepseek_client),
) -> StreamingResponse:
    async def generate():
        parents: list[dict] = []
        retrieval_warning = ""
        if settings.rag_enabled:
            yield sse("status", {"stage": "retrieving", "message": "正在检索本班知识资料"})
            try:
                result = await run_in_threadpool(retrieve_with_rerank, request.question.strip(), user.class_id)
                parents = result["parents"]
            except RetrievalError as exc:
                retrieval_warning = str(exc)
        system_prompt = rag_system_prompt(user.role, parents)
        messages = [{"role": "system", "content": system_prompt}] + [item.model_dump() for item in request.history[-10:]] + [{"role": "user", "content": request.question.strip()}]
        if parents:
            yield sse("sources", {"items": [
                {"label": item["source_label"], "filename": item["filename"], "heading": item["section_path"] or item["heading"], "page": item["page"]}
                for item in parents
            ]})
        elif retrieval_warning:
            yield sse("status", {"stage": "retrieval_warning", "message": f"知识库暂不可用，将按通用问题回答：{retrieval_warning}"})
        yield sse("status", {"stage": "generating", "message": "DeepSeek正在生成回答"})
        try:
            async for content in client.stream(messages):
                yield sse("content", {"text": content})
            yield sse("done", {"ok": True})
        except DeepSeekError as exc:
            yield sse("error", {"message": str(exc)})

    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/status")
async def ai_status() -> dict[str, str | bool]:
    return {
        "configured": bool(os.getenv("DEEPSEEK_API_KEY", "").strip()),
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    }


@router.post("/student-qa", response_model=StudentAnswerResponse)
async def student_qa(
    request: StudentQuestionRequest,
    client: DeepSeekClient = Depends(get_deepseek_client),
) -> StudentAnswerResponse:
    messages = [
        {
            "role": "system",
            "content": (
                "你是高校团务问答助手。请用简洁、友善的中文回答。当前没有接入学校本地知识库，"
                "不得捏造政策、文件名称、截止日期或引用来源；遇到校级差异时，明确建议咨询本班团支书或学院老师。"
            ),
        },
        {"role": "user", "content": request.question.strip()},
    ]
    try:
        answer = await client.complete(messages)
    except DeepSeekError as exc:
        raise ai_http_error(exc) from exc
    return StudentAnswerResponse(answer=answer)


@router.post("/meeting-summary", response_model=MeetingSummaryResponse)
async def meeting_summary(
    request: MeetingSummaryRequest,
    user: User = Depends(require_secretary),
    client: DeepSeekClient = Depends(get_deepseek_client),
) -> MeetingSummaryResponse:
    transcript, was_redacted = redact_sensitive_text(request.transcript.strip())
    messages = [
        {
            "role": "system",
            "content": (
                "你是高校团务会议纪要助手。只根据提供的文字稿整理，不得补写未出现的事实。"
                "返回一个 JSON 对象，字段必须为 title、summary、key_points、decisions、action_items。"
                "key_points 和 decisions 是字符串数组；action_items 是对象数组，每项含 task、owner、deadline。"
                "未明确负责人或截止日期时填写‘未指定’。"
            ),
        },
        {"role": "user", "content": f"会议类型：{request.meeting_type}\n\n文字稿：\n{transcript}"},
    ]
    try:
        raw_result = await client.complete(messages, json_output=True)
        data = json.loads(raw_result)
        data.update(
            meeting_type=request.meeting_type,
            requires_manual_review=True,
            redacted_sensitive_data=was_redacted,
        )
        return MeetingSummaryResponse.model_validate(data)
    except DeepSeekError as exc:
        raise ai_http_error(exc) from exc
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="AI 返回格式不完整，请重新生成") from exc
