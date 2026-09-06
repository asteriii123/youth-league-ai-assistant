"""
文件管理模块。

负责：

1. 文件上传保存
2. 文件路径管理
3. 文件删除
4. 音视频文件管理


服务：

知识库上传

会议音视频上传

生成文件管理。
"""
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile

from app.core.config import settings


MAX_NOTICE_FILE_SIZE = 20 * 1024 * 1024
NOTICE_SUFFIXES = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".jpg", ".jpeg", ".png"}
NOTICE_MIME_TYPES = {
    "application/pdf", "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "image/jpeg", "image/png", "application/octet-stream",
}


def save_notice_attachment(upload: UploadFile) -> tuple[str, str]:
    original_name = Path(upload.filename or "附件").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in NOTICE_SUFFIXES or upload.content_type not in NOTICE_MIME_TYPES:
        raise HTTPException(status_code=400, detail="附件仅支持PDF、Word、Excel和常见图片")
    target_dir = settings.uploads_dir / "notices"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{uuid4().hex}{suffix}"
    size = 0
    with target.open("wb") as output:
        while chunk := upload.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_NOTICE_FILE_SIZE:
                output.close(); target.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="附件不能超过20MB")
            output.write(chunk)
    return str(target), original_name


def delete_notice_attachment(path: str | None) -> None:
    if not path:
        return
    target = Path(path).resolve()
    allowed_dir = (settings.uploads_dir / "notices").resolve()
    if allowed_dir not in target.parents:
        return
    target.unlink(missing_ok=True)


def save_submission_attachment(upload: UploadFile) -> tuple[str, str]:
    original_name = Path(upload.filename or "材料").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in NOTICE_SUFFIXES or upload.content_type not in NOTICE_MIME_TYPES:
        raise HTTPException(status_code=400, detail="材料仅支持PDF、Word、Excel和常见图片")
    target_dir = settings.uploads_dir / "submissions"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{uuid4().hex}{suffix}"
    size = 0
    with target.open("wb") as output:
        while chunk := upload.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_NOTICE_FILE_SIZE:
                output.close(); target.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="材料不能超过20MB")
            output.write(chunk)
    return str(target), original_name


def delete_submission_attachment(path: str | None) -> None:
    if not path:
        return
    target = Path(path).resolve()
    allowed_dir = (settings.uploads_dir / "submissions").resolve()
    if allowed_dir in target.parents:
        target.unlink(missing_ok=True)


MEETING_SUFFIXES = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".mp4", ".mov", ".avi", ".mkv", ".webm"}
MEETING_MIME_PREFIXES = ("audio/", "video/")
MAX_MEETING_FILE_SIZE = 500 * 1024 * 1024


def save_meeting_media(upload: UploadFile) -> tuple[str, str, str]:
    original_name = Path(upload.filename or "会议文件").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in MEETING_SUFFIXES or not (upload.content_type or "").startswith(MEETING_MIME_PREFIXES):
        raise HTTPException(status_code=400, detail="仅支持常见音频或视频文件")
    target_dir = settings.uploads_dir / "meetings"
    target_dir.mkdir(parents=True, exist_ok=True)
    upload_id = uuid4().hex
    target = target_dir / f"{upload_id}{suffix}"
    size = 0
    with target.open("wb") as output:
        while chunk := upload.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_MEETING_FILE_SIZE:
                output.close(); target.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="音视频文件不能超过500MB")
            output.write(chunk)
    return str(target), original_name, upload_id


def resolve_meeting_upload(upload_id: str) -> Path | None:
    if not upload_id or not upload_id.isalnum():
        return None
    allowed_dir = (settings.uploads_dir / "meetings").resolve()
    matches = list(allowed_dir.glob(f"{upload_id}.*"))
    return matches[0].resolve() if len(matches) == 1 and allowed_dir in matches[0].resolve().parents else None


def delete_meeting_media(path: str | None) -> None:
    if not path:
        return
    target = Path(path).resolve()
    allowed_dir = (settings.uploads_dir / "meetings").resolve()
    if allowed_dir in target.parents:
        target.unlink(missing_ok=True)


KNOWLEDGE_SUFFIXES = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".txt"}
KNOWLEDGE_MIME_TYPES = {
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "text/plain",
    "application/octet-stream",
}
MAX_KNOWLEDGE_FILE_SIZE = 50 * 1024 * 1024


def knowledge_file_type(suffix: str) -> str:
    if suffix in {".doc", ".docx"}:
        return "word"
    if suffix in {".ppt", ".pptx"}:
        return "ppt"
    return suffix.lstrip(".")


def save_knowledge_file(upload: UploadFile) -> tuple[str, str, str]:
    original_name = Path(upload.filename or "资料").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in KNOWLEDGE_SUFFIXES or upload.content_type not in KNOWLEDGE_MIME_TYPES:
        raise HTTPException(status_code=400, detail="知识资料仅支持PDF、Word、PPT和TXT")
    target_dir = settings.uploads_dir / "knowledge"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{uuid4().hex}{suffix}"
    size = 0
    with target.open("wb") as output:
        while chunk := upload.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_KNOWLEDGE_FILE_SIZE:
                output.close()
                target.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="知识资料不能超过50MB")
            output.write(chunk)
    return str(target), original_name, knowledge_file_type(suffix)


def delete_knowledge_file(path: str | None) -> None:
    if not path:
        return
    target = Path(path).resolve()
    allowed_dir = (settings.uploads_dir / "knowledge").resolve()
    if allowed_dir in target.parents:
        target.unlink(missing_ok=True)
