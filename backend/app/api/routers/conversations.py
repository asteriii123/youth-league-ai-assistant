"""
聊天会话管理模块。

负责：

1. 创建聊天窗口
2. 保存聊天历史
3. 管理用户消息
4. 连接AI回答


作用：

保存用户和AI之间的对话记录。


数据：

ChatConversation
ChatMessage


属于AI聊天系统的数据层接口。
"""

import asyncio
import json
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.api.routers.ai import sse
from app.core.config import settings
from app.core.database import SessionLocal, get_db
from app.core.security import get_current_user
from app.llm.deepseek import DeepSeekClient, DeepSeekError, get_deepseek_client
from app.llm.prompts import rag_system_prompt
from app.models.entities import ChatConversation, ChatMessage, MeetingJob, User
from app.rag.retrieval import RetrievalError, retrieve_with_rerank
from app.search.service import search_web, web_sources_prompt


router = APIRouter(prefix="/api/ai/conversations", tags=["AI对话"])


class ConversationUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=80)


class QuestionPayload(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    web_search_enabled: bool = False


CASUAL_MESSAGES = {
    "你好", "您好", "嗨", "哈喽", "hello", "hi", "谢谢", "感谢", "再见", "拜拜",
    "你是谁", "你能做什么", "早上好", "中午好", "下午好", "晚上好",
}
FOLLOW_UP_PREFIXES = ("那", "那么", "这个", "这些", "上述", "刚才", "还需要", "还有", "为什么", "具体")


def is_casual_message(question: str) -> bool:
    normalized = question.strip().lower().rstrip("!！?？。,.，~～ ")
    return normalized in CASUAL_MESSAGES


def retrieval_query(question: str, history: list[ChatMessage]) -> str:
    if question.startswith(FOLLOW_UP_PREFIXES):
        previous = next((item.content for item in reversed(history) if item.role == "user"), "")
        if previous:
            return f"{previous}\n追问：{question}"
    return question


def owned_conversation(db: Session, conversation_id: int, user: User) -> ChatConversation:
    conversation = db.get(ChatConversation, conversation_id)
    if not conversation or conversation.user_id != user.id or conversation.class_id != user.class_id:
        raise HTTPException(status_code=404, detail="对话不存在")
    return conversation


def serialize_conversation(item: ChatConversation) -> dict[str, Any]:
    return {
        "id": item.id, "title": item.title, "mode": item.mode,
        "created_at": item.created_at.isoformat(), "updated_at": item.updated_at.isoformat(),
    }


def serialize_message(item: ChatMessage) -> dict[str, Any]:
    return {
        "id": item.id, "role": item.role, "content": item.content, "status": item.status,
        "sources": json.loads(item.sources_json or "[]"), "meeting_job_id": item.meeting_job_id,
        "created_at": item.created_at.isoformat(),
    }


@router.get("")
def list_conversations(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[dict]:
    items = db.scalars(
        select(ChatConversation).where(ChatConversation.user_id == user.id).order_by(ChatConversation.updated_at.desc())
    ).all()
    return [serialize_conversation(item) for item in items]


@router.post("", status_code=201)
def create_conversation(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    item = ChatConversation(user_id=user.id, class_id=user.class_id, title="新对话", mode="assistant")
    db.add(item); db.commit(); db.refresh(item)
    return serialize_conversation(item)


@router.patch("/{conversation_id}")
def rename_conversation(conversation_id: int, payload: ConversationUpdate, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    item = owned_conversation(db, conversation_id, user)
    item.title = payload.title.strip(); item.updated_at = datetime.now()
    db.commit(); db.refresh(item)
    return serialize_conversation(item)


@router.delete("/{conversation_id}", status_code=204)
def remove_conversation(conversation_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> None:
    item = owned_conversation(db, conversation_id, user)
    db.query(MeetingJob).filter(MeetingJob.conversation_id == item.id).update({"conversation_id": None})
    db.execute(delete(ChatMessage).where(ChatMessage.conversation_id == item.id))
    db.delete(item); db.commit()


@router.get("/{conversation_id}/messages")
def list_messages(conversation_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[dict]:
    item = owned_conversation(db, conversation_id, user)
    messages = db.scalars(select(ChatMessage).where(ChatMessage.conversation_id == item.id).order_by(ChatMessage.id)).all()
    return [serialize_message(message) for message in messages]


@router.post("/{conversation_id}/messages/stream")
async def send_message(
    conversation_id: int,
    payload: QuestionPayload,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    client: DeepSeekClient = Depends(get_deepseek_client),
) -> StreamingResponse:
    conversation = owned_conversation(db, conversation_id, user)
    question = payload.question.strip()
    existing_count = db.scalar(select(ChatMessage.id).where(ChatMessage.conversation_id == conversation.id).limit(1))
    if not existing_count:
        conversation.title = question[:30]
    conversation.updated_at = datetime.now()
    user_message = ChatMessage(conversation_id=conversation.id, role="user", content=question, status="complete")
    assistant_message = ChatMessage(conversation_id=conversation.id, role="assistant", content="", status="streaming")
    db.add_all([user_message, assistant_message]); db.commit(); db.refresh(user_message); db.refresh(assistant_message)
    assistant_id = assistant_message.id

    async def generate():
        parents: list[dict] = []
        web_results: list[dict] = []
        sources: list[dict] = []
        answer = ""
        try:
            with SessionLocal() as history_db:
                history_rows = history_db.scalars(
                    select(ChatMessage).where(
                        ChatMessage.conversation_id == conversation_id,
                        ChatMessage.status == "complete",
                        ChatMessage.id < user_message.id,
                    ).order_by(ChatMessage.id.desc()).limit(10)
                ).all()
            history_rows = list(reversed(history_rows))
            history = [{"role": item.role, "content": item.content} for item in history_rows]
            if settings.rag_enabled and not is_casual_message(question):
                yield sse("status", {"stage": "retrieving", "message": "正在检索本班知识资料"})
                try:
                    query = retrieval_query(question, history_rows)
                    result = await run_in_threadpool(retrieve_with_rerank, query, user.class_id)
                    parents = [
                        item for item in result["parents"]
                        if float(item.get("rerank_score", 0)) >= settings.rag_min_rerank_score
                    ]
                    for index, item in enumerate(parents, start=1):
                        item["source_label"] = f"资料{index}"
                except RetrievalError as exc:
                    yield sse("status", {"stage": "retrieval_warning", "message": f"知识库暂不可用：{exc}"})
            web_context = ""
            if payload.web_search_enabled and not is_casual_message(question):
                yield sse("status", {"stage": "web_searching", "message": "正在智能搜索互联网"})
                response = await search_web(retrieval_query(question, history_rows))
                web_results = response.results
                web_context = web_sources_prompt(web_results)
                if response.warnings and not web_results:
                    yield sse("status", {"stage": "web_search_warning", "message": "智能搜索暂不可用，将继续生成回答"})
            sources = [
                {"type": "knowledge", "label": item["source_label"], "filename": item["filename"], "heading": item["section_path"] or item["heading"], "page": item["page"]}
                for item in parents
            ]
            sources.extend([
                {"type": "web", "label": item["source_label"], "title": item["title"], "url": item["url"],
                 "domain": urlsplit(item["url"]).netloc, "provider": item["provider"]}
                for item in web_results
            ])
            if sources:
                yield sse("sources", {"items": sources})
            messages = [{"role": "system", "content": rag_system_prompt(user.role, parents) + web_context}, *history, {"role": "user", "content": question}]
            yield sse("message", {"user_message_id": user_message.id, "assistant_message_id": assistant_id})
            yield sse("status", {"stage": "generating", "message": "DeepSeek正在生成回答"})
            async for content in client.stream(messages):
                answer += content
                yield sse("content", {"text": content})
            with SessionLocal() as result_db:
                saved = result_db.get(ChatMessage, assistant_id)
                if saved:
                    saved.content = answer; saved.status = "complete"; saved.sources_json = json.dumps(sources, ensure_ascii=False)
                    result_db.commit()
            yield sse("done", {"ok": True})
        except DeepSeekError as exc:
            with SessionLocal() as result_db:
                saved = result_db.get(ChatMessage, assistant_id)
                if saved:
                    saved.content = str(exc); saved.status = "failed"; result_db.commit()
            yield sse("error", {"message": str(exc)})
        except asyncio.CancelledError:
            with SessionLocal() as result_db:
                saved = result_db.get(ChatMessage, assistant_id)
                if saved:
                    saved.content = answer; saved.status = "stopped"; saved.sources_json = json.dumps(sources, ensure_ascii=False)
                    result_db.commit()
            raise
        except Exception as exc:
            message = str(exc) or "聊天处理失败"
            with SessionLocal() as result_db:
                saved = result_db.get(ChatMessage, assistant_id)
                if saved:
                    saved.content = message; saved.status = "failed"; result_db.commit()
            yield sse("error", {"message": message})

    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
