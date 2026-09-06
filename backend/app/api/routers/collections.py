"""
资料收藏模块。

负责：

1. 用户收藏资料
2. 查看收藏列表
3. 删除收藏


用于：

学生保存重要文件。

例如：

- 入党流程
- 团务文件
- 通知模板
"""
import csv
import io
import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.files import delete_submission_attachment, save_submission_attachment
from app.core.security import get_current_user, require_secretary
from app.models.entities import CollectionSubmission, CollectionTask, User


router = APIRouter(prefix="/api/collections", tags=["信息收集"])
TASK_STATUSES = {"draft", "published", "closed"}
FIELD_TYPES = {"text", "date", "single"}


class TaskPayload(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=5000)
    fields: list[dict] = Field(min_length=1, max_length=30)
    status: str = "draft"
    deadline: str | None = None
    attachment_required: bool = False
    allow_modify: bool = True


class ReturnPayload(BaseModel):
    reason: str = Field(min_length=1, max_length=1000)


def validate_fields(fields: list[dict]) -> list[dict]:
    clean: list[dict] = []
    ids: set[str] = set()
    for index, item in enumerate(fields, start=1):
        field_id = str(item.get("id", "")).strip()
        label = str(item.get("label", "")).strip()
        field_type = str(item.get("type", ""))
        if not field_id or field_id in ids or not label or field_type not in FIELD_TYPES:
            raise HTTPException(status_code=400, detail=f"第{index}个字段配置无效")
        options = [str(value).strip() for value in item.get("options", []) if str(value).strip()]
        if field_type == "single" and len(options) < 2:
            raise HTTPException(status_code=400, detail=f"“{label}”至少需要两个选项")
        ids.add(field_id)
        clean.append({"id": field_id, "label": label[:100], "type": field_type, "required": bool(item.get("required")), "options": options})
    return clean


def owned_task(db: Session, task_id: int, user: User) -> CollectionTask:
    task = db.get(CollectionTask, task_id)
    if not task or task.class_id != user.class_id:
        raise HTTPException(status_code=404, detail="收集任务不存在")
    return task


def task_dict(task: CollectionTask, db: Session, user: User) -> dict:
    submissions = db.scalars(select(CollectionSubmission).where(CollectionSubmission.task_id == task.id)).all()
    mine = next((item for item in submissions if item.student_id == user.id), None)
    students = db.scalars(select(User).where(User.class_id == task.class_id, User.role == "student")).all()
    return {
        "id": task.id, "title": task.title, "description": task.description,
        "fields": json.loads(task.fields_json), "status": task.status, "deadline": task.deadline,
        "attachment_required": task.attachment_required, "allow_modify": task.allow_modify,
        "submitted_count": sum(item.status == "submitted" for item in submissions),
        "student_count": len(students), "my_status": mine.status if mine else None,
        "created_at": task.created_at.isoformat(), "updated_at": task.updated_at.isoformat(),
    }


def submission_dict(item: CollectionSubmission, student: User) -> dict:
    return {
        "id": item.id, "student_id": student.id, "student_name": student.display_name,
        "answers": json.loads(item.answers_json), "status": item.status,
        "return_reason": item.return_reason, "attachment_name": item.attachment_name,
        "submitted_at": item.submitted_at.isoformat() if item.submitted_at else None,
        "updated_at": item.updated_at.isoformat(),
    }


@router.get("")
def list_tasks(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[dict]:
    statement = select(CollectionTask).where(CollectionTask.class_id == user.class_id)
    if user.role == "student":
        statement = statement.where(CollectionTask.status == "published")
    tasks = db.scalars(statement.order_by(CollectionTask.created_at.desc())).all()
    return [task_dict(item, db, user) for item in tasks]


@router.post("", status_code=201)
def create_task(payload: TaskPayload, user: User = Depends(require_secretary), db: Session = Depends(get_db)) -> dict:
    if payload.status not in {"draft", "published"}:
        raise HTTPException(status_code=400, detail="新任务只能保存为草稿或发布")
    task = CollectionTask(
        class_id=user.class_id, author_id=user.id, title=payload.title.strip(), description=payload.description.strip(),
        fields_json=json.dumps(validate_fields(payload.fields), ensure_ascii=False), status=payload.status,
        deadline=payload.deadline or None, attachment_required=payload.attachment_required, allow_modify=payload.allow_modify,
        requires_file=payload.attachment_required, allow_update=payload.allow_modify,
    )
    db.add(task); db.commit(); db.refresh(task)
    return task_dict(task, db, user)


@router.patch("/{task_id}")
def update_task(task_id: int, payload: TaskPayload, user: User = Depends(require_secretary), db: Session = Depends(get_db)) -> dict:
    task = owned_task(db, task_id, user)
    if payload.status not in TASK_STATUSES:
        raise HTTPException(status_code=400, detail="任务状态无效")
    task.title = payload.title.strip(); task.description = payload.description.strip()
    task.fields_json = json.dumps(validate_fields(payload.fields), ensure_ascii=False)
    task.status = payload.status; task.deadline = payload.deadline or None
    task.attachment_required = payload.attachment_required; task.allow_modify = payload.allow_modify
    task.requires_file = payload.attachment_required; task.allow_update = payload.allow_modify
    db.commit(); db.refresh(task)
    return task_dict(task, db, user)


@router.delete("/{task_id}", status_code=204)
def delete_task(task_id: int, user: User = Depends(require_secretary), db: Session = Depends(get_db)) -> None:
    task = owned_task(db, task_id, user)
    submissions = db.scalars(select(CollectionSubmission).where(CollectionSubmission.task_id == task.id)).all()
    for item in submissions:
        delete_submission_attachment(item.attachment_path)
        db.delete(item)
    db.delete(task); db.commit()


@router.get("/{task_id}/submissions")
def list_submissions(task_id: int, user: User = Depends(require_secretary), db: Session = Depends(get_db)) -> dict:
    task = owned_task(db, task_id, user)
    students = db.scalars(select(User).where(User.class_id == task.class_id, User.role == "student").order_by(User.display_name)).all()
    records = {item.student_id: item for item in db.scalars(select(CollectionSubmission).where(CollectionSubmission.task_id == task.id)).all()}
    submitted = [submission_dict(records[item.id], item) for item in students if item.id in records]
    missing = [{"student_id": item.id, "student_name": item.display_name} for item in students if item.id not in records]
    overdue: list[dict] = []
    try:
        deadline_passed = bool(task.deadline and datetime.fromisoformat(task.deadline) < datetime.now())
    except ValueError:
        deadline_passed = False
    if deadline_passed:
        overdue = [{"student_id": item.id, "student_name": item.display_name} for item in students if item.id not in records or records[item.id].status != "submitted"]
    return {"submissions": submitted, "missing": missing, "overdue": overdue}


@router.post("/{task_id}/submissions/{submission_id}/return")
def return_submission(task_id: int, submission_id: int, payload: ReturnPayload, user: User = Depends(require_secretary), db: Session = Depends(get_db)) -> dict:
    owned_task(db, task_id, user)
    item = db.get(CollectionSubmission, submission_id)
    if not item or item.task_id != task_id:
        raise HTTPException(status_code=404, detail="提交记录不存在")
    item.status = "returned"; item.return_reason = payload.reason.strip()
    db.commit()
    return {"status": "returned"}


@router.get("/{task_id}/my-submission")
def my_submission(task_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict | None:
    task = owned_task(db, task_id, user)
    if user.role != "student" or task.status != "published":
        raise HTTPException(status_code=403, detail="无法查看该任务")
    item = db.scalar(select(CollectionSubmission).where(CollectionSubmission.task_id == task.id, CollectionSubmission.student_id == user.id))
    return submission_dict(item, user) if item else None


@router.put("/{task_id}/my-submission")
def save_my_submission(
    task_id: int, answers: str = Form(...), submit: bool = Form(False), remove_attachment: bool = Form(False),
    attachment: UploadFile | None = File(None), user: User = Depends(get_current_user), db: Session = Depends(get_db),
) -> dict:
    task = owned_task(db, task_id, user)
    if user.role != "student" or task.status != "published":
        raise HTTPException(status_code=403, detail="无法提交该任务")
    item = db.scalar(select(CollectionSubmission).where(CollectionSubmission.task_id == task.id, CollectionSubmission.student_id == user.id))
    if item and item.status == "submitted" and not task.allow_modify:
        raise HTTPException(status_code=409, detail="该任务提交后不允许修改")
    try:
        answer_data = json.loads(answers)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="表单答案格式错误") from exc
    if not isinstance(answer_data, dict):
        raise HTTPException(status_code=400, detail="表单答案格式错误")
    fields = json.loads(task.fields_json)
    for field in fields:
        value = str(answer_data.get(field["id"], "")).strip()
        if submit and field["required"] and not value:
            raise HTTPException(status_code=400, detail=f"请填写“{field['label']}”")
        if value and field["type"] == "single" and value not in field["options"]:
            raise HTTPException(status_code=400, detail=f"“{field['label']}”选项无效")
    if not item:
        item = CollectionSubmission(task_id=task.id, student_id=user.id)
        db.add(item)
    if remove_attachment or attachment:
        delete_submission_attachment(item.attachment_path); item.attachment_path = None; item.attachment_name = None
    if attachment:
        item.attachment_path, item.attachment_name = save_submission_attachment(attachment)
    if submit and task.attachment_required and not item.attachment_path:
        raise HTTPException(status_code=400, detail="该任务要求上传附件")
    item.answers_json = json.dumps(answer_data, ensure_ascii=False)
    item.status = "submitted" if submit else "draft"; item.return_reason = None if submit else item.return_reason
    item.submitted_at = datetime.now() if submit else item.submitted_at
    db.commit(); db.refresh(item)
    return submission_dict(item, user)


@router.get("/{task_id}/submissions/{submission_id}/attachment")
def download_submission(task_id: int, submission_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> FileResponse:
    task = owned_task(db, task_id, user)
    item = db.get(CollectionSubmission, submission_id)
    if not item or item.task_id != task.id or (user.role == "student" and item.student_id != user.id):
        raise HTTPException(status_code=404, detail="附件不存在")
    if not item.attachment_path or not Path(item.attachment_path).is_file():
        raise HTTPException(status_code=404, detail="附件不存在")
    return FileResponse(item.attachment_path, filename=item.attachment_name)


@router.get("/{task_id}/export")
def export_submissions(task_id: int, user: User = Depends(require_secretary), db: Session = Depends(get_db)) -> StreamingResponse:
    task = owned_task(db, task_id, user)
    fields = json.loads(task.fields_json)
    students = {item.id: item for item in db.scalars(select(User).where(User.class_id == task.class_id, User.role == "student")).all()}
    output = io.StringIO(); output.write("\ufeff")
    writer = csv.writer(output); writer.writerow(["姓名", *[item["label"] for item in fields], "状态", "提交时间", "附件"])
    for item in db.scalars(select(CollectionSubmission).where(CollectionSubmission.task_id == task.id)).all():
        answers = json.loads(item.answers_json); student = students.get(item.student_id)
        writer.writerow([student.display_name if student else "未知学生", *[answers.get(field["id"], "") for field in fields], item.status, item.submitted_at.isoformat() if item.submitted_at else "", item.attachment_name or ""])
    content = output.getvalue().encode("utf-8")
    headers = {"Content-Disposition": f"attachment; filename=collection-{task.id}.csv"}
    return StreamingResponse(iter([content]), media_type="text/csv; charset=utf-8", headers=headers)
