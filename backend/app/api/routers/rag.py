"""
RAG检索测试接口。

负责：

1. 接收用户查询
2. 调用RAG检索系统
3. 返回相关知识片段


主要用途：

开发测试知识库搜索效果。


实际聊天场景：

ai.py
    ↓
retrieval.py


本文件主要用于调试RAG。
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.security import require_secretary
from app.models.entities import User
from app.rag.retrieval import RetrievalError, retrieve_with_rerank


router = APIRouter(prefix="/api/rag", tags=["RAG"])


class SearchPayload(BaseModel):
    query: str = Field(min_length=2, max_length=1000)


@router.post("/search/debug")
def search_debug(payload: SearchPayload, user: User = Depends(require_secretary)) -> dict:
    try:
        return retrieve_with_rerank(payload.query.strip(), user.class_id)
    except RetrievalError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
