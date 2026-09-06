"""
项目后端启动入口。

主要功能：
1. 创建 FastAPI 应用实例
2. 注册跨域 CORS
3. 加载数据库初始化逻辑
4. 注册所有 API 路由
5. 提供基础健康检查接口

调用关系：

前端 Vue
    |
    ↓
main.py
    |
    ↓
各个 router 接口

这是整个后端服务的入口文件。
"""
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers.ai import router as ai_router
from app.api.routers.auth import router as auth_router
from app.api.routers.collections import router as collections_router
from app.api.routers.conversations import router as conversations_router
from app.api.routers.knowledge import router as knowledge_router
from app.api.routers.meeting_agent import router as meeting_agent_router
from app.api.routers.meetings import router as meetings_router
from app.api.routers.notices import router as notices_router
from app.api.routers.rag import router as rag_router
from app.core.init_db import initialize_database


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    yield


app = FastAPI(
    title="团支书 AI 助手 API",
    description="项目第一阶段的 FastAPI 基础服务。",
    version="0.1.0",
    lifespan=lifespan,
)

allowed_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
frontend_origin = os.getenv("FRONTEND_ORIGIN", "").strip().rstrip("/")
if frontend_origin:
    allowed_origins.append(frontend_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ai_router)
app.include_router(auth_router)
app.include_router(notices_router)
app.include_router(collections_router)
app.include_router(meetings_router)
app.include_router(knowledge_router)
app.include_router(rag_router)
app.include_router(conversations_router)
app.include_router(meeting_agent_router)


@app.get("/", tags=["基础"])
async def root() -> dict[str, str]:
    return {"message": "团支书 AI 助手后端正在运行"}


@app.get("/api/health", tags=["基础"])
async def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "league-secretary-ai-assistant"}


@app.get("/api/welcome", tags=["演示"])
async def welcome(
    role: Literal["secretary", "student"] = Query(default="student"),
) -> dict[str, str]:
    role_name = "团支书" if role == "secretary" else "学生"
    return {
        "message": f"你好，{role_name}！前端已经成功连接 FastAPI。",
        "role": role,
    }
