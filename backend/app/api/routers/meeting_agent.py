"""
会议AI Agent任务管理模块。

负责：

1. 创建会议AI任务
2. 调用LangGraph会议工作流
3. 管理任务状态
4. 人工审核转写结果
5. 人工确认会议纪要


流程：

上传会议文件
    ↓
MeetingJob
    ↓
LangGraph Agent
    ↓
Whisper
    ↓
DeepSeek总结
    ↓
生成会议纪要


属于会议自动化核心入口。
"""
import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.agents.meeting import run_meeting_graph
from app.core.database import get_db
from app.core.files import save_meeting_media
from app.core.security import require_secretary
from app.models.entities import ChatConversation, ChatMessage, MeetingJob, User


router = APIRouter(prefix="/api/ai/meeting-jobs", tags=["会议Agent"])
DEFAULT_INSTRUCTION = "请整理为标准会议纪要"


class TranscriptReview(BaseModel):
    transcript: str = Field(min_length=20, max_length=100000)


class MinutesReview(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    meeting_type: str = Field(min_length=1, max_length=30)
    summary: str = Field(min_length=1, max_length=10000)
    key_points: list[str] = Field(default_factory=list, max_length=100)
    decisions: list[str] = Field(default_factory=list, max_length=100)
    action_items: list[dict] = Field(default_factory=list, max_length=100)
    requires_manual_review: bool = True
    redacted_sensitive_data: bool = True


def owned_job(db: Session, job_id: int, user: User) -> MeetingJob:
    job = db.get(MeetingJob, job_id)
    if not job or job.user_id != user.id or job.class_id != user.class_id:
        raise HTTPException(status_code=404, detail="会议任务不存在")
    return job


def serialize(job: MeetingJob) -> dict:
    return {
        "id": job.id, "conversation_id": job.conversation_id, "meeting_type": job.meeting_type,
        "instruction": job.instruction,
        "status": job.status, "source_name": job.source_name, "transcript": job.transcript,
        "filtered_transcript": job.filtered_transcript,
        "minutes": json.loads(job.minutes_json or "{}"), "meeting_record_id": job.meeting_record_id,
        "download_ready": bool(job.document_path and Path(job.document_path).is_file()),
        "error_message": job.error_message, "created_at": job.created_at.isoformat(), "updated_at": job.updated_at.isoformat(),
    }


def infer_meeting_type(instruction: str) -> str:
    for name in ("主题团日", "团课", "支部会议"):
        if name in instruction:
            return name
    return "其他"


@router.post("", status_code=202)
def create_job(
    background_tasks: BackgroundTasks,
    conversation_id: int = Form(...),
    instruction: str = Form(default=DEFAULT_INSTRUCTION),
    file: UploadFile = File(...),
    user: User = Depends(require_secretary),
    db: Session = Depends(get_db),
) -> dict:
    conversation = db.get(ChatConversation, conversation_id)
    if not conversation or conversation.user_id != user.id or conversation.class_id != user.class_id:
        raise HTTPException(status_code=404, detail="对话不存在")
    cleaned_instruction = instruction.strip()[:2000] or DEFAULT_INSTRUCTION
    meeting_type = infer_meeting_type(cleaned_instruction)
    path, original_name, _ = save_meeting_media(file)
    job = MeetingJob(
        conversation_id=conversation.id, user_id=user.id, class_id=user.class_id,
        meeting_type=meeting_type, instruction=cleaned_instruction,
        source_path=path, source_name=original_name, status="queued",
    )
    db.add(job); db.flush()
    db.add_all([
        ChatMessage(conversation_id=conversation.id, role="user", content=f"{cleaned_instruction}\n\n附件：{original_name}", status="complete", meeting_job_id=job.id),
        ChatMessage(conversation_id=conversation.id, role="assistant", content="正在本地转写音视频……", status="processing", meeting_job_id=job.id),
    ])
    if conversation.title == "新对话":
        conversation.title = f"会议：{Path(original_name).stem[:24]}"
    conversation.updated_at = datetime.now(); db.commit(); db.refresh(job)
    background_tasks.add_task(run_meeting_graph, job.id)
    return serialize(job)


@router.get("/{job_id}")
def get_job(job_id: int, user: User = Depends(require_secretary), db: Session = Depends(get_db)) -> dict:
    return serialize(owned_job(db, job_id, user))


@router.post("/{job_id}/resume-transcript", status_code=202)
def resume_transcript(job_id: int, payload: TranscriptReview, background_tasks: BackgroundTasks, user: User = Depends(require_secretary), db: Session = Depends(get_db)) -> dict:
    job = owned_job(db, job_id, user)
    if job.status != "awaiting_transcript_review":
        raise HTTPException(status_code=409, detail="当前任务不在转写稿确认阶段")
    background_tasks.add_task(run_meeting_graph, job.id, {"transcript": payload.transcript.strip()})
    job.status = "filtering"; job.error_message = None; db.commit(); db.refresh(job)
    return serialize(job)


@router.post("/{job_id}/confirm-minutes", status_code=202)
def confirm_minutes(job_id: int, payload: MinutesReview, background_tasks: BackgroundTasks, user: User = Depends(require_secretary), db: Session = Depends(get_db)) -> dict:
    job = owned_job(db, job_id, user)
    if job.status != "awaiting_minutes_review":
        raise HTTPException(status_code=409, detail="当前任务不在纪要确认阶段")
    background_tasks.add_task(run_meeting_graph, job.id, payload.model_dump())
    job.status = "creating_document"; job.error_message = None; db.commit(); db.refresh(job)
    return serialize(job)


@router.post("/{job_id}/retry", status_code=202)
def retry_job(job_id: int, background_tasks: BackgroundTasks, user: User = Depends(require_secretary), db: Session = Depends(get_db)) -> dict:
    job = owned_job(db, job_id, user)
    if job.status != "failed":
        raise HTTPException(status_code=409, detail="只能重试失败的任务")
    job.status = "queued"; job.error_message = None; db.commit(); db.refresh(job)
    background_tasks.add_task(run_meeting_graph, job.id, None, True)
    return serialize(job)


@router.get("/{job_id}/document")
def download_document(job_id: int, user: User = Depends(require_secretary), db: Session = Depends(get_db)) -> FileResponse:
    job = owned_job(db, job_id, user)
    target = Path(job.document_path or "")
    if not job.document_path or not target.is_file():
        raise HTTPException(status_code=404, detail="Word文档尚未生成")
    return FileResponse(target, filename=f"{Path(job.source_name).stem}-会议纪要.docx", media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
