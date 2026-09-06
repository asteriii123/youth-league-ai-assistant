"""
知识文档处理模块。

负责：

1. 文件解析
2. 文本提取
3. 文档切片
4. 父子Chunk生成
5. 创建知识索引


流程：

PDF/Word
 ↓
文本
 ↓
Chunk
 ↓
Embedding
 ↓
向量库


属于RAG数据生产流程。
"""
import hashlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from app.core.config import settings
from app.core.database import SessionLocal
from app.models.entities import KnowledgeChunk, KnowledgeDocument


class KnowledgeError(Exception):
    pass


# ---- Small-to-Big 分块参数（中文字符数） ----
PARENT_MIN = 800
PARENT_TARGET = 1500
PARENT_MAX = 2000
SMALL_MIN = 200
SMALL_TARGET = 300
SMALL_MAX = 350
SMALL_OVERLAP = 50

_SENTENCE_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;\n]?")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


@dataclass
class Block:
    type: str            # heading / paragraph / table / list
    text: str
    level: int = 0       # 标题层级，非标题为 0
    page: int = 1


def compute_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def split_sentences(text: str) -> list[str]:
    return [part for part in _SENTENCE_RE.findall(text) if part.strip()]


def _split_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def _hard_split(text: str, limit: int) -> list[str]:
    """把超长文本硬切成不超过 limit 的片段，尽量在标点处断开。"""
    pieces: list[str] = []
    while len(text) > limit:
        cut = text[:limit]
        match = list(re.finditer(r"[。！？!?；;\n]", cut))
        if match and match[-1].end() > limit * 0.5:
            pieces.append(text[: match[-1].end()])
            text = text[match[-1].end():]
            continue
        pieces.append(text[:limit])
        text = text[limit:]
    if text:
        pieces.append(text)
    return pieces


def build_parent_chunks(blocks: list[Block]) -> list[dict]:
    """按标题与自然段生成父块，目标 800～1500 字，最大 2000 字。"""
    chunks: list[dict] = []
    buffer: list[str] = []
    section_stack: list[tuple[int, str]] = []
    page = 1

    def heading() -> str:
        return section_stack[-1][1] if section_stack else ""

    def section_path() -> str:
        return " > ".join(text for _, text in section_stack)

    def flush() -> None:
        nonlocal buffer
        if buffer:
            content = "".join(buffer).strip()
            chunks.append({"content": content, "heading": heading(), "section_path": section_path(), "page": page, "char_count": len(content)})
            buffer = []

    for block in blocks:
        if block.type == "heading":
            if sum(len(t) for t in buffer) >= PARENT_MIN:
                flush()
            section_stack = [(level, text) for level, text in section_stack if level < block.level]
            section_stack.append((block.level, block.text))
            page = block.page
            buffer.append(block.text + "\n")
        else:
            text = block.text.strip()
            if not text:
                continue
            if not buffer:
                page = block.page
            if len(text) > PARENT_MAX:
                if buffer:
                    flush()
                for piece in _hard_split(text, PARENT_MAX):
                    chunks.append({"content": piece.strip(), "heading": heading(), "section_path": section_path(), "page": page, "char_count": len(piece.strip())})
                continue
            if sum(len(t) for t in buffer) + len(text) > PARENT_MAX:
                flush()
            buffer.append(text + "\n")
            if sum(len(t) for t in buffer) >= PARENT_TARGET:
                flush()

    flush()
    return chunks


def build_small_chunks(parent_text: str) -> list[str]:
    """从父块切分小块，目标 200～350 字，重叠约 50 字。"""
    sentences = split_sentences(parent_text)
    chunks: list[str] = []
    buffer = ""
    for sentence in sentences:
        if len(sentence) > SMALL_MAX:
            if buffer.strip():
                chunks.append(buffer.strip())
                buffer = ""
            chunks.extend(piece.strip() for piece in _hard_split(sentence, SMALL_MAX))
            continue
        if buffer and len(buffer) + len(sentence) > SMALL_MAX:
            chunks.append(buffer.strip())
            buffer = buffer[-SMALL_OVERLAP:] if len(buffer) > SMALL_OVERLAP else buffer
        buffer += sentence
    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks


# ---- 文档解析（重依赖按需导入） ----

def convert_office_to_pdf(source: Path) -> Path:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        default = Path(r"C:\Program Files\LibreOffice\program\soffice.exe")
        if default.is_file():
            soffice = str(default)
    if not soffice:
        raise KnowledgeError("未检测到LibreOffice，安装后才能处理Word和PPT文件")
    out_dir = settings.converted_dir / "knowledge"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(out_dir), str(source)],
            check=True, capture_output=True, timeout=600,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise KnowledgeError("LibreOffice无法转换文件，请检查文件是否完整") from exc
    pdf_path = out_dir / f"{source.stem}.pdf"
    if not pdf_path.is_file():
        raise KnowledgeError("LibreOffice转换后未生成PDF文件")
    return pdf_path


def _pdf_has_text_layer(pdf_path: Path) -> bool:
    try:
        import pymupdf as fitz  # PyMuPDF
    except ImportError as exc:
        raise KnowledgeError("缺少PyMuPDF，无法检测PDF类型，请安装后端依赖") from exc
    document = fitz.open(str(pdf_path))
    try:
        return any(page.get_text().strip() for page in document)
    finally:
        document.close()


def _extract_with_docling(pdf_path: Path) -> list[Block]:
    """用 Docling 解析电子 PDF，尽力保留标题层级与页码；失败返回空列表以触发回退。"""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        return []
    try:
        result = DocumentConverter().convert(str(pdf_path))
        document = result.document
    except Exception:
        return []
    blocks: list[Block] = []
    current_page = 1
    try:
        for item, _level in document.iterate_items():
            page_no = getattr(item, "page_no", None)
            if page_no is None and getattr(item, "prov", None):
                prov = item.prov[0] if isinstance(item.prov, list) and item.prov else item.prov
                page_no = getattr(prov, "page_no", None)
            if page_no:
                current_page = int(page_no)
            text = ""
            if getattr(item, "text", None):
                text = str(item.text)
            text = text.strip()
            if not text:
                continue
            label_name = str(getattr(item, "label", "")).lower()
            if "title" in label_name or "heading" in label_name or "header" in label_name:
                blocks.append(Block("heading", text, level=2, page=current_page))
            else:
                blocks.append(Block("paragraph", text, page=current_page))
    except Exception:
        return []
    return blocks


def _extract_with_pymupdf(pdf_path: Path) -> list[Block]:
    """逐页提取 PDF 文本，保留页码；用字号启发式识别标题。"""
    import pymupdf as fitz  # PyMuPDF

    document = fitz.open(str(pdf_path))
    blocks: list[Block] = []
    try:
        for page_index, page in enumerate(document, start=1):
            for block in page.get_text("dict").get("blocks", []):
                if block.get("type") != 0:
                    continue
                spans = [span for line in block.get("lines", []) for span in line.get("spans", [])]
                if not spans:
                    continue
                text = "".join(span.get("text", "") for span in spans).strip()
                if not text:
                    continue
                max_size = max(span.get("size", 0) for span in spans)
                if max_size >= 15:
                    blocks.append(Block("heading", text, level=2, page=page_index))
                else:
                    blocks.append(Block("paragraph", text, page=page_index))
    finally:
        document.close()
    return blocks


def _extract_pdf_scanned(pdf_path: Path) -> list[Block]:
    try:
        import pymupdf as fitz
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise KnowledgeError("扫描PDF需要PyMuPDF、pytesseract和Pillow，请安装后端依赖") from exc
    tesseract_cmd = shutil.which("tesseract") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if not Path(tesseract_cmd).is_file():
        raise KnowledgeError("未检测到Tesseract，安装后才能解析扫描PDF")
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    os.environ.setdefault("TESSDATA_PREFIX", str(settings.tessdata_dir.resolve()))
    document = fitz.open(str(pdf_path))
    blocks: list[Block] = []
    try:
        for page_index, page in enumerate(document, start=1):
            pix = page.get_pixmap(dpi=200)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(image, lang="chi_sim")
            if text.strip():
                blocks.extend(Block("paragraph", para, page=page_index) for para in _split_paragraphs(text))
    finally:
        document.close()
    return blocks


def _extract_txt(path: Path) -> list[Block]:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks: list[Block] = []
    for raw in text.split("\n\n"):
        line = raw.strip()
        if not line:
            continue
        match = _HEADING_RE.match(line)
        if match:
            blocks.append(Block("heading", match.group(2).strip(), level=len(match.group(1)), page=1))
        else:
            blocks.append(Block("paragraph", line, page=1))
    return blocks


def extract_blocks(source: Path, file_type: str) -> list[Block]:
    if file_type == "txt":
        return _extract_txt(source)
    pdf_path = source
    temporary = False
    if file_type in {"word", "ppt"}:
        pdf_path = convert_office_to_pdf(source)
        temporary = True
    try:
        if _pdf_has_text_layer(pdf_path):
            return _extract_with_docling(pdf_path) or _extract_with_pymupdf(pdf_path)
        return _extract_pdf_scanned(pdf_path)
    finally:
        if temporary:
            pdf_path.unlink(missing_ok=True)


def _estimate_page_count(blocks: list[Block]) -> int:
    return max((block.page for block in blocks), default=1)


def _clear_document_chunks(db, document_id: int) -> None:
    smalls = db.scalars(select(KnowledgeChunk).where(KnowledgeChunk.document_id == document_id, KnowledgeChunk.parent_id.is_not(None))).all()
    parents = db.scalars(select(KnowledgeChunk).where(KnowledgeChunk.document_id == document_id, KnowledgeChunk.parent_id.is_(None))).all()
    for chunk in smalls:
        db.delete(chunk)
    for chunk in parents:
        db.delete(chunk)
    db.flush()


def process_knowledge_document(document_id: int) -> None:
    """解析并分块一个知识文档，结果与状态写回 SQLite。独立会话，供后台任务调用。"""
    with SessionLocal() as db:
        document = db.get(KnowledgeDocument, document_id)
        if not document:
            return
        document.status = "processing"
        document.error_message = None
        db.commit()
        try:
            source = Path(document.stored_path)
            if not source.is_file():
                raise KnowledgeError("原始文件已丢失，请重新上传")
            blocks = extract_blocks(source, document.file_type)
            if not blocks:
                raise KnowledgeError("未能从文件中解析出可用的文本内容")
            parents = build_parent_chunks(blocks)
            if not parents:
                raise KnowledgeError("分块后没有可用内容")
            _clear_document_chunks(db, document.id)
            order = 0
            parent_count = 0
            small_count = 0
            for parent in parents:
                order += 1
                parent_count += 1
                parent_chunk = KnowledgeChunk(
                    document_id=document.id, chunk_type="parent", content=parent["content"],
                    heading=parent["heading"], section_path=parent["section_path"],
                    page=parent["page"], char_count=parent["char_count"], order_index=order,
                )
                db.add(parent_chunk)
                db.flush()
                for small_text in build_small_chunks(parent["content"]):
                    order += 1
                    small_count += 1
                    db.add(KnowledgeChunk(
                        document_id=document.id, parent_id=parent_chunk.id, chunk_type="small",
                        content=small_text, heading=parent["heading"], section_path=parent["section_path"],
                        page=parent["page"], char_count=len(small_text), order_index=order,
                    ))
            document.page_count = _estimate_page_count(blocks)
            document.parent_count = parent_count
            document.small_count = small_count
            document.status = "done"
            document.error_message = None
            document.index_status = "pending"
            document.index_error = None
            db.commit()
            if settings.rag_enabled:
                from app.rag.retrieval import RetrievalError, index_document
                try:
                    index_document(document.id)
                except RetrievalError:
                    pass
        except KnowledgeError as exc:
            document.status = "failed"
            document.error_message = str(exc)
            db.commit()
        except Exception as exc:  # 兜底，避免后台任务静默失败
            document.status = "failed"
            document.error_message = f"解析失败：{exc}"
            db.commit()
