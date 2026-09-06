"""
通知管理模块。

负责：

1. 发布通知
2. 查看通知
3. 班级通知管理


应用场景：

团支书发布：

- 团费通知
- 会议通知
- 工作安排


未来可以接入：

飞书机器人Agent。
"""
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.files import delete_notice_attachment, save_notice_attachment
from app.core.security import get_current_user, require_secretary
from app.llm.deepseek import DeepSeekClient, DeepSeekError, get_deepseek_client
from app.models.entities import Notice, NoticeRead, User


router = APIRouter(prefix="/api/notices", tags=["通知"])
VALID_STATUSES = {"draft", "published", "withdrawn"}


class DraftRequest(BaseModel):
    topic: str = Field(min_length=2, max_length=500)


def get_owned_notice(db: Session, notice_id: int, user: User) -> Notice:
    notice = db.get(Notice, notice_id)
    if not notice or notice.class_id != user.class_id:
        raise HTTPException(status_code=404, detail="通知不存在")
    return notice


def serialize(notice: Notice, db: Session, user: User) -> dict:
    read_count = db.scalar(select(func.count()).select_from(NoticeRead).where(NoticeRead.notice_id == notice.id)) or 0
    student_count = db.scalar(select(func.count()).select_from(User).where(User.class_id == notice.class_id, User.role == "student")) or 0
    is_read = bool(db.scalar(select(NoticeRead.id).where(NoticeRead.notice_id == notice.id, NoticeRead.user_id == user.id)))
    return {
        "id": notice.id, "title": notice.title, "content": notice.content, "status": notice.status,
        "deadline": notice.deadline, "attachment_name": notice.attachment_name,
        "read_count": read_count, "student_count": student_count, "is_read": is_read,
        "created_at": notice.created_at.isoformat(), "updated_at": notice.updated_at.isoformat(),
    }


@router.get("")
def list_notices(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[dict]:
    statement = select(Notice).where(Notice.class_id == user.class_id)
    if user.role == "student":
        statement = statement.where(Notice.status == "published")
    notices = db.scalars(statement.order_by(Notice.created_at.desc())).all()
    return [serialize(item, db, user) for item in notices]


@router.post("/ai-draft")
async def ai_draft(request: DraftRequest, user: User = Depends(require_secretary), client: DeepSeekClient = Depends(get_deepseek_client)) -> dict:
    messages = [
        {"role": "system", "content": "你是高校班级团支书的通知写作助手。输出简洁正式的中文通知，包含标题和正文；不得编造具体日期、地点或学校政策，缺失信息用【请填写】标记。"},
        {"role": "user", "content": request.topic.strip()},
    ]
    try:
        text = await client.complete(messages)
    except DeepSeekError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    title = lines[0].removeprefix("标题：").removeprefix("# ") if lines else "团务通知"
    content = "\n".join(lines[1:]) if len(lines) > 1 else text
    return {"title": title[:200], "content": content}


@router.post("", status_code=201)
def create_notice(
    title: str = Form(..., min_length=1, max_length=200), content: str = Form(..., min_length=1),
    status: str = Form("draft"), deadline: str | None = Form(None), attachment: UploadFile | None = File(None),
    user: User = Depends(require_secretary), db: Session = Depends(get_db),
) -> dict:
    if status not in {"draft", "published"}:
        raise HTTPException(status_code=400, detail="新通知只能保存为草稿或发布")
    path, name = save_notice_attachment(attachment) if attachment else (None, None)
    notice = Notice(class_id=user.class_id, author_id=user.id, title=title.strip(), content=content.strip(), status=status, deadline=deadline or None, attachment_path=path, attachment_name=name)
    db.add(notice); db.commit(); db.refresh(notice)
    return serialize(notice, db, user)


@router.patch("/{notice_id}")
def update_notice(
    notice_id: int, title: str = Form(..., min_length=1, max_length=200), content: str = Form(..., min_length=1),
    status: str = Form(...), deadline: str | None = Form(None), attachment: UploadFile | None = File(None),
    remove_attachment: bool = Form(False), user: User = Depends(require_secretary), db: Session = Depends(get_db),
) -> dict:
    notice = get_owned_notice(db, notice_id, user)
    if status not in VALID_STATUSES:
        raise HTTPException(status_code=400, detail="通知状态无效")
    notice.title, notice.content, notice.status, notice.deadline = title.strip(), content.strip(), status, deadline or None
    if remove_attachment or attachment:
        delete_notice_attachment(notice.attachment_path); notice.attachment_path = None; notice.attachment_name = None
    if attachment:
        notice.attachment_path, notice.attachment_name = save_notice_attachment(attachment)
    db.commit(); db.refresh(notice)
    return serialize(notice, db, user)


@router.delete("/{notice_id}", status_code=204)
def delete_notice(notice_id: int, user: User = Depends(require_secretary), db: Session = Depends(get_db)) -> None:
    notice = get_owned_notice(db, notice_id, user)
    delete_notice_attachment(notice.attachment_path)
    db.query(NoticeRead).filter(NoticeRead.notice_id == notice_id).delete()
    db.delete(notice); db.commit()


@router.post("/{notice_id}/read")
def mark_read(notice_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    notice = get_owned_notice(db, notice_id, user)
    if user.role != "student" or notice.status != "published":
        raise HTTPException(status_code=403, detail="该通知不能标记为已读")
    if not db.scalar(select(NoticeRead.id).where(NoticeRead.notice_id == notice_id, NoticeRead.user_id == user.id)):
        db.add(NoticeRead(notice_id=notice_id, user_id=user.id)); db.commit()
    return {"status": "read"}


@router.get("/{notice_id}/readers")
def notice_readers(notice_id: int, user: User = Depends(require_secretary), db: Session = Depends(get_db)) -> dict:
    notice = get_owned_notice(db, notice_id, user)
    students = db.scalars(select(User).where(User.class_id == notice.class_id, User.role == "student").order_by(User.display_name)).all()
    read_ids = set(db.scalars(select(NoticeRead.user_id).where(NoticeRead.notice_id == notice_id)).all())
    return {
        "read": [{"id": item.id, "display_name": item.display_name} for item in students if item.id in read_ids],
        "unread": [{"id": item.id, "display_name": item.display_name} for item in students if item.id not in read_ids],
    }


@router.get("/{notice_id}/attachment")
def download_attachment(notice_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> FileResponse:
    notice = get_owned_notice(db, notice_id, user)
    if user.role == "student" and notice.status != "published":
        raise HTTPException(status_code=404, detail="附件不存在")
    if not notice.attachment_path or not Path(notice.attachment_path).is_file():
        raise HTTPException(status_code=404, detail="附件不存在")
    return FileResponse(notice.attachment_path, filename=notice.attachment_name)
