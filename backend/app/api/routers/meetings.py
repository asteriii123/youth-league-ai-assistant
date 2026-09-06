"""
会议记录管理模块。

负责：

1. 音视频转文字
2. 保存会议记录
3. 修改会议纪要
4. 删除会议记录
5. 生成Word会议文档


数据来源：

会议Agent生成结果。


流程：

音视频
 ↓
Whisper转写
 ↓
AI总结
 ↓
保存MeetingRecord
 ↓
生成docx
"""
import json

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.meeting import build_minutes_docx
from app.core.config import settings
from app.core.database import get_db
from app.core.files import delete_meeting_media, resolve_meeting_upload, save_meeting_media
from app.core.security import require_secretary
from app.models.entities import MeetingRecord, User
from app.services.transcription import TranscriptionError, transcribe_media


router = APIRouter(prefix="/api/meetings", tags=["会议助手"])


class ActionItemPayload(BaseModel):
    task: str = Field(min_length=1, max_length=500)
    owner: str = Field(default="未指定", max_length=100)
    deadline: str = Field(default="未指定", max_length=100)


class MeetingPayload(BaseModel):
    meeting_type: str = Field(min_length=1, max_length=30)
    title: str = Field(min_length=1, max_length=200)
    transcript: str = Field(min_length=20, max_length=100000)
    summary: str = Field(min_length=1, max_length=10000)
    key_points: list[str] = Field(default_factory=list, max_length=100)
    decisions: list[str] = Field(default_factory=list, max_length=100)
    action_items: list[ActionItemPayload] = Field(default_factory=list, max_length=100)
    upload_id: str | None = None
    source_name: str | None = Field(default=None, max_length=255)


def owned_record(db: Session, record_id: int, user: User) -> MeetingRecord:
    record = db.get(MeetingRecord, record_id)
    if not record or record.class_id != user.class_id:
        raise HTTPException(status_code=404, detail="会议纪要不存在")
    return record


def serialize(record: MeetingRecord) -> dict:
    return {
        "id": record.id, "meeting_type": record.meeting_type, "title": record.title,
        "transcript": record.transcript, "summary": record.summary,
        "key_points": json.loads(record.key_points_json), "decisions": json.loads(record.decisions_json),
        "action_items": json.loads(record.action_items_json), "source_name": record.source_name,
        "created_at": record.created_at.isoformat(), "updated_at": record.updated_at.isoformat(),
    }


@router.post("/transcribe")
def transcribe(file: UploadFile = File(...), user: User = Depends(require_secretary)) -> dict:
    path, original_name, upload_id = save_meeting_media(file)
    try:
        source = resolve_meeting_upload(upload_id)
        if not source:
            raise TranscriptionError("上传文件保存失败，请重新上传")
        transcript = transcribe_media(source)
    except TranscriptionError as exc:
        delete_meeting_media(path)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"transcript": transcript, "upload_id": upload_id, "source_name": original_name}


@router.get("")
def list_records(user: User = Depends(require_secretary), db: Session = Depends(get_db)) -> list[dict]:
    records = db.scalars(select(MeetingRecord).where(MeetingRecord.class_id == user.class_id).order_by(MeetingRecord.created_at.desc())).all()
    return [serialize(item) for item in records]


@router.post("", status_code=201)
def create_record(payload: MeetingPayload, user: User = Depends(require_secretary), db: Session = Depends(get_db)) -> dict:
    source = resolve_meeting_upload(payload.upload_id) if payload.upload_id else None
    if payload.upload_id and not source:
        raise HTTPException(status_code=400, detail="上传的音视频文件已失效，请重新转写")
    record = MeetingRecord(
        class_id=user.class_id, author_id=user.id, meeting_type=payload.meeting_type, title=payload.title.strip(),
        transcript=payload.transcript.strip(), summary=payload.summary.strip(),
        summary_json=json.dumps({"summary": payload.summary}, ensure_ascii=False),
        key_points_json=json.dumps(payload.key_points, ensure_ascii=False), decisions_json=json.dumps(payload.decisions, ensure_ascii=False),
        action_items_json=json.dumps([item.model_dump() for item in payload.action_items], ensure_ascii=False),
        source_path=str(source) if source else None, source_name=payload.source_name if source else None,
    )
    db.add(record); db.commit(); db.refresh(record)
    return serialize(record)


@router.patch("/{record_id}")
def update_record(record_id: int, payload: MeetingPayload, user: User = Depends(require_secretary), db: Session = Depends(get_db)) -> dict:
    record = owned_record(db, record_id, user)
    record.meeting_type = payload.meeting_type; record.title = payload.title.strip(); record.transcript = payload.transcript.strip(); record.summary = payload.summary.strip()
    record.summary_json = json.dumps({"summary": payload.summary}, ensure_ascii=False)
    record.key_points_json = json.dumps(payload.key_points, ensure_ascii=False); record.decisions_json = json.dumps(payload.decisions, ensure_ascii=False)
    record.action_items_json = json.dumps([item.model_dump() for item in payload.action_items], ensure_ascii=False)
    db.commit(); db.refresh(record)
    return serialize(record)


@router.delete("/{record_id}", status_code=204)
def delete_record(record_id: int, user: User = Depends(require_secretary), db: Session = Depends(get_db)) -> None:
    record = owned_record(db, record_id, user)
    delete_meeting_media(record.source_path)
    (settings.meeting_documents_dir / f"meeting-record-{record.id}.docx").unlink(missing_ok=True)
    db.delete(record); db.commit()


@router.get("/{record_id}/document")
def download_record_document(record_id: int, user: User = Depends(require_secretary), db: Session = Depends(get_db)) -> FileResponse:
    record = owned_record(db, record_id, user)
    minutes = {
        "title": record.title, "meeting_type": record.meeting_type, "summary": record.summary,
        "key_points": json.loads(record.key_points_json), "decisions": json.loads(record.decisions_json),
        "action_items": json.loads(record.action_items_json),
    }
    target = settings.meeting_documents_dir / f"meeting-record-{record.id}.docx"
    build_minutes_docx(target, minutes, record.transcript)
    return FileResponse(target, filename=f"{record.title}.docx", media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
