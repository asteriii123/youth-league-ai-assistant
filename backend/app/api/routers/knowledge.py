"""
知识库管理接口。

负责：

1. 上传知识文件
2. 查看知识资料
3. 删除知识资料
4. 重新解析文件
5. 建立向量索引
6. 测试RAG搜索


支持文件：

- PDF
- Word
- PPT
- TXT


流程：

上传文件
    ↓
文档解析
    ↓
文本切片
    ↓
Embedding
    ↓
Chroma向量数据库


属于RAG知识库入口。
"""
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.files import delete_knowledge_file, save_knowledge_file
from app.core.security import require_secretary
from app.models.entities import KnowledgeChunk, KnowledgeDocument, User
from app.rag.documents import compute_file_hash, process_knowledge_document
from app.rag.retrieval import RetrievalError, delete_document_index, hybrid_search, index_document_safe, set_document_enabled


router = APIRouter(prefix="/api/knowledge", tags=["知识资料"])


class EnabledPayload(BaseModel):
    enabled: bool


class SearchPayload(BaseModel):
    query: str = Field(min_length=2, max_length=1000)


def owned_document(db: Session, document_id: int, user: User) -> KnowledgeDocument:
    document = db.get(KnowledgeDocument, document_id)
    if not document or document.class_id != user.class_id:
        raise HTTPException(status_code=404, detail="知识资料不存在")
    return document


def serialize_document(document: KnowledgeDocument) -> dict:
    return {
        "id": document.id, "filename": document.filename, "file_type": document.file_type,
        "status": document.status, "error_message": document.error_message,
        "page_count": document.page_count, "parent_count": document.parent_count,
        "small_count": document.small_count, "enabled": document.enabled,
        "index_status": document.index_status, "index_error": document.index_error,
        "indexed_at": document.indexed_at.isoformat() if document.indexed_at else None,
        "created_at": document.created_at.isoformat(), "updated_at": document.updated_at.isoformat(),
    }


@router.post("", status_code=201)
def upload_document(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    user: User = Depends(require_secretary),
    db: Session = Depends(get_db),
) -> dict:
    path, original_name, file_type = save_knowledge_file(file)
    file_hash = compute_file_hash(Path(path))
    duplicate = db.scalar(select(KnowledgeDocument).where(
        KnowledgeDocument.class_id == user.class_id, KnowledgeDocument.file_hash == file_hash,
    ))
    if duplicate:
        delete_knowledge_file(path)
        raise HTTPException(status_code=409, detail="该文件已上传并建立索引，无需重复处理")
    document = KnowledgeDocument(
        class_id=user.class_id, author_id=user.id, filename=original_name,
        file_type=file_type, file_hash=file_hash, stored_path=path, status="pending",
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    background.add_task(process_knowledge_document, document.id)
    return serialize_document(document)


@router.get("")
def list_documents(user: User = Depends(require_secretary), db: Session = Depends(get_db)) -> list[dict]:
    documents = db.scalars(select(KnowledgeDocument).where(KnowledgeDocument.class_id == user.class_id).order_by(KnowledgeDocument.created_at.desc())).all()
    return [serialize_document(item) for item in documents]


@router.get("/{document_id}")
def get_document(document_id: int, user: User = Depends(require_secretary), db: Session = Depends(get_db)) -> dict:
    document = owned_document(db, document_id, user)
    chunks = db.scalars(select(KnowledgeChunk).where(KnowledgeChunk.document_id == document.id).order_by(KnowledgeChunk.order_index)).all()
    parents = [item for item in chunks if item.chunk_type == "parent"]
    smalls = [item for item in chunks if item.chunk_type == "small"]
    smalls_by_parent: dict[int, list[dict]] = {}
    for item in smalls:
        smalls_by_parent.setdefault(item.parent_id, []).append({"id": item.id, "content": item.content, "char_count": item.char_count})
    parent_payload = [
        {
            "id": item.id, "content": item.content, "heading": item.heading,
            "section_path": item.section_path, "page": item.page, "char_count": item.char_count,
            "smalls": smalls_by_parent.get(item.id, []),
        }
        for item in parents
    ]
    return {**serialize_document(document), "parents": parent_payload}


@router.post("/{document_id}/retry")
def retry_document(
    document_id: int, background: BackgroundTasks,
    user: User = Depends(require_secretary), db: Session = Depends(get_db),
) -> dict:
    document = owned_document(db, document_id, user)
    if document.status == "processing":
        raise HTTPException(status_code=409, detail="该资料正在解析中")
    document.status = "pending"
    document.error_message = None
    db.commit()
    db.refresh(document)
    background.add_task(process_knowledge_document, document.id)
    return serialize_document(document)


@router.post("/{document_id}/reindex")
def reindex_document(
    document_id: int, background: BackgroundTasks,
    user: User = Depends(require_secretary), db: Session = Depends(get_db),
) -> dict:
    document = owned_document(db, document_id, user)
    if document.status != "done":
        raise HTTPException(status_code=409, detail="资料尚未解析完成")
    if document.index_status == "indexing":
        raise HTTPException(status_code=409, detail="资料正在建立索引")
    document.index_status = "pending"; document.index_error = None; db.commit(); db.refresh(document)
    background.add_task(index_document_safe, document.id)
    return serialize_document(document)


@router.post("/search/debug")
def search_debug(payload: SearchPayload, user: User = Depends(require_secretary)) -> dict:
    try:
        return hybrid_search(payload.query.strip(), user.class_id)
    except RetrievalError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.patch("/{document_id}/enabled")
def toggle_document(document_id: int, payload: EnabledPayload, user: User = Depends(require_secretary), db: Session = Depends(get_db)) -> dict:
    document = owned_document(db, document_id, user)
    document.enabled = payload.enabled
    db.commit()
    db.refresh(document)
    if document.index_status == "indexed":
        try:
            set_document_enabled(document.id, document.class_id, document.enabled)
        except RetrievalError as exc:
            document.index_status = "failed"; document.index_error = str(exc); db.commit(); db.refresh(document)
    return serialize_document(document)


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: int, user: User = Depends(require_secretary), db: Session = Depends(get_db)) -> None:
    document = owned_document(db, document_id, user)
    delete_knowledge_file(document.stored_path)
    chunks = db.scalars(select(KnowledgeChunk).where(KnowledgeChunk.document_id == document.id)).all()
    for chunk in chunks:
        db.delete(chunk)
    class_id = document.class_id
    db.delete(document)
    db.commit()
    try:
        delete_document_index(document_id, class_id)
    except RetrievalError:
        pass
