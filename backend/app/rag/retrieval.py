"""
RAG检索核心模块。

负责：

1. Embedding生成
2. Chroma向量搜索
3. BM25关键词搜索
4. RRF融合
5. Rerank排序
6. 返回最终知识片段


核心流程：

Query

 ↓

Embedding

 ↓

Vector Search

+

BM25

 ↓

RRF

 ↓

Rerank

 ↓

Top-K结果


这是知识库的大脑。
"""
import json
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import jieba
from rank_bm25 import BM25Okapi
from sqlalchemy import select

from app.core.config import settings
from app.core.database import SessionLocal
from app.models.entities import KnowledgeChunk, KnowledgeDocument


class RetrievalError(Exception):
    pass


class ModelScopeEmbeddingClient:
    def embed(self, texts: list[str]) -> list[list[float]]:
        if not settings.modelscope_api_token:
            raise RetrievalError("MODELSCOPE_API_TOKEN尚未配置，无法建立向量索引")
        last_error: Exception | None = None
        for attempt in range(settings.embedding_retries):
            try:
                response = httpx.post(
                    f"{settings.modelscope_base_url}/embeddings",
                    headers={"Authorization": f"Bearer {settings.modelscope_api_token}"},
                    json={"model": settings.embedding_model, "input": texts, "encoding_format": "float"}, timeout=120,
                )
                response.raise_for_status()
                items = sorted(response.json()["data"], key=lambda item: item.get("index", 0))
                embeddings = [item["embedding"] for item in items]
                if len(embeddings) != len(texts):
                    raise ValueError("Embedding数量与输入不一致")
                return embeddings
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt + 1 < settings.embedding_retries:
                    time.sleep(1.5 * (attempt + 1))
        raise RetrievalError("魔搭Embedding调用失败，请检查Token、模型名和网络") from last_error


class LocalEmbeddingClient:
    _model = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            if LocalEmbeddingClient._model is None:
                from sentence_transformers import SentenceTransformer
                LocalEmbeddingClient._model = SentenceTransformer(
                    settings.embedding_model,
                    device=settings.embedding_device,
                    cache_folder=str(settings.models_dir),
                )
            vectors = LocalEmbeddingClient._model.encode(
                texts,
                batch_size=max(1, settings.embedding_batch_size),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return vectors.tolist()
        except ImportError as exc:
            raise RetrievalError("尚未安装本地Embedding依赖，请执行pip install -r requirements.txt") from exc
        except Exception as exc:
            raise RetrievalError("本地Embedding运行失败，请检查模型下载、磁盘空间和网络") from exc


def get_embedding_client():
    if settings.embedding_provider == "local":
        return LocalEmbeddingClient()
    if settings.embedding_provider == "modelscope":
        return ModelScopeEmbeddingClient()
    raise RetrievalError("EMBEDDING_PROVIDER只支持local或modelscope")


class ModelScopeRerankClient:
    def rerank(self, query: str, documents: list[str]) -> list[dict]:
        if not settings.modelscope_api_token:
            raise RetrievalError("MODELSCOPE_API_TOKEN尚未配置，无法进行Rerank")
        last_error: Exception | None = None
        for attempt in range(settings.embedding_retries):
            try:
                response = httpx.post(
                    f"{settings.modelscope_base_url}/rerank",
                    headers={"Authorization": f"Bearer {settings.modelscope_api_token}"},
                    json={
                        "model": settings.rerank_model,
                        "query": query,
                        "documents": documents,
                        "top_n": len(documents),
                        "return_documents": False,
                    },
                    timeout=120,
                )
                response.raise_for_status()
                payload = response.json()
                items = payload.get("results", payload.get("data", []))
                ranked = [
                    {
                        "index": int(item["index"]),
                        "score": float(item.get("relevance_score", item.get("score"))),
                    }
                    for item in items
                ]
                if not ranked:
                    raise ValueError("Rerank返回空结果")
                return sorted(ranked, key=lambda item: item["score"], reverse=True)
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt + 1 < settings.embedding_retries:
                    time.sleep(1.5 * (attempt + 1))
        raise RetrievalError("魔搭Rerank调用失败，请检查Token、模型名和网络") from last_error


class LocalRerankClient:
    _model = None

    def rerank(self, query: str, documents: list[str]) -> list[dict]:
        try:
            if LocalRerankClient._model is None:
                from sentence_transformers import CrossEncoder
                LocalRerankClient._model = CrossEncoder(
                    settings.rerank_model,
                    device=settings.rerank_device,
                    cache_folder=str(settings.models_dir),
                )
            scores = LocalRerankClient._model.predict(
                [(query, document) for document in documents],
                batch_size=max(1, settings.rerank_batch_size),
                show_progress_bar=False,
            )
            ranked = [{"index": index, "score": float(score)} for index, score in enumerate(scores)]
            return sorted(ranked, key=lambda item: item["score"], reverse=True)
        except ImportError as exc:
            raise RetrievalError("尚未安装本地Rerank依赖，请执行pip install -r requirements.txt") from exc
        except Exception as exc:
            raise RetrievalError("本地Rerank运行失败，请检查模型下载、磁盘空间和内存") from exc


def get_rerank_client():
    if settings.rerank_provider == "local":
        return LocalRerankClient()
    if settings.rerank_provider == "modelscope":
        return ModelScopeRerankClient()
    raise RetrievalError("RERANK_PROVIDER只支持local或modelscope")


def get_collection():
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        identity = f"{settings.embedding_provider}:{settings.embedding_model}".encode("utf-8")
        collection_name = f"knowledge_small_chunks_{hashlib.sha1(identity).hexdigest()[:10]}"
        return client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})
    except Exception as exc:
        raise RetrievalError("Chroma本地向量库初始化失败") from exc


def tokenize(text: str) -> list[str]:
    return [token.strip().lower() for token in jieba.lcut_for_search(text) if token.strip()]


def bm25_path(class_id: int) -> Path:
    return settings.indexes_dir / f"bm25-class-{class_id}.json"


def rebuild_bm25(class_id: int) -> None:
    with SessionLocal() as db:
        rows = db.execute(
            select(KnowledgeChunk, KnowledgeDocument)
            .join(KnowledgeDocument, KnowledgeDocument.id == KnowledgeChunk.document_id)
            .where(
                KnowledgeDocument.class_id == class_id, KnowledgeDocument.enabled.is_(True),
                KnowledgeDocument.index_status == "indexed", KnowledgeChunk.chunk_type == "small",
            ).order_by(KnowledgeChunk.id)
        ).all()
    entries = [{
        "chunk_id": chunk.id, "document_id": document.id, "parent_id": chunk.parent_id,
        "content": chunk.content, "filename": document.filename, "heading": chunk.heading,
        "section_path": chunk.section_path, "page": chunk.page, "tokens": tokenize(chunk.content),
    } for chunk, document in rows]
    path = bm25_path(class_id); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "entries": entries}, ensure_ascii=False), encoding="utf-8")


def index_document(document_id: int, embedder=None, collection=None) -> None:
    embedder = embedder or get_embedding_client()
    with SessionLocal() as db:
        document = db.get(KnowledgeDocument, document_id)
        if not document or document.status != "done":
            return
        document.index_status = "indexing"; document.index_error = None; db.commit()
        chunks = db.scalars(select(KnowledgeChunk).where(KnowledgeChunk.document_id == document.id, KnowledgeChunk.chunk_type == "small").order_by(KnowledgeChunk.id)).all()
        try:
            store = collection or get_collection()
            if not chunks:
                raise RetrievalError("没有可用于检索的小块")
            store.delete(where={"document_id": document.id})
            batch_size = max(1, settings.embedding_batch_size)
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start:start + batch_size]
                embeddings = embedder.embed([item.content for item in batch])
                store.upsert(
                    ids=[f"chunk-{item.id}" for item in batch], embeddings=embeddings,
                    documents=[item.content for item in batch],
                    metadatas=[{
                        "chunk_id": item.id, "document_id": document.id, "class_id": document.class_id,
                        "parent_id": item.parent_id or 0, "filename": document.filename,
                        "heading": item.heading, "section_path": item.section_path, "page": item.page,
                        "enabled": bool(document.enabled),
                    } for item in batch],
                )
            document.index_status = "indexed"; document.index_error = None; document.indexed_at = datetime.now(); db.commit()
            rebuild_bm25(document.class_id)
        except Exception as exc:
            document.index_status = "failed"; document.index_error = str(exc); db.commit()
            if isinstance(exc, RetrievalError):
                raise
            raise RetrievalError("知识资料索引失败") from exc


def index_document_safe(document_id: int) -> None:
    try:
        index_document(document_id)
    except RetrievalError:
        pass


def delete_document_index(document_id: int, class_id: int, collection=None) -> None:
    try:
        (collection or get_collection()).delete(where={"document_id": document_id})
    finally:
        rebuild_bm25(class_id)


def set_document_enabled(document_id: int, class_id: int, enabled: bool, collection=None) -> None:
    try:
        store = collection or get_collection()
        result = store.get(where={"document_id": document_id}, include=["metadatas"])
        if result.get("ids"):
            metadatas = [{**metadata, "enabled": enabled} for metadata in result["metadatas"]]
            store.update(ids=result["ids"], metadatas=metadatas)
    finally:
        rebuild_bm25(class_id)


def vector_search(query: str, class_id: int, top_k: int | None = None, embedder=None, collection=None) -> list[dict]:
    top_k = top_k or settings.rag_recall_top_k
    vector = (embedder or get_embedding_client()).embed([query])[0]
    result = (collection or get_collection()).query(
        query_embeddings=[vector], n_results=top_k,
        where={"$and": [{"class_id": {"$eq": class_id}}, {"enabled": {"$eq": True}}]},
        include=["documents", "metadatas", "distances"],
    )
    output: list[dict] = []
    for rank, (content, metadata, distance) in enumerate(zip(result["documents"][0], result["metadatas"][0], result["distances"][0]), start=1):
        output.append({**metadata, "content": content, "rank": rank, "score": round(1 - float(distance), 6)})
    return output


def bm25_search(query: str, class_id: int, top_k: int | None = None) -> list[dict]:
    top_k = top_k or settings.rag_recall_top_k
    path = bm25_path(class_id)
    if not path.is_file():
        rebuild_bm25(class_id)
    entries = json.loads(path.read_text(encoding="utf-8"))["entries"]
    if not entries:
        return []
    model = BM25Okapi([item["tokens"] for item in entries])
    scores = model.get_scores(tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)
    return [{**entries[index], "rank": rank, "score": round(float(score), 6)} for rank, (index, score) in enumerate(ranked[:top_k], start=1) if score > 0]


def rrf_fuse(vector_results: list[dict], bm25_results: list[dict], k: int | None = None, top_k: int | None = None) -> list[dict]:
    k = k or settings.rag_rrf_k; top_k = top_k or settings.rag_fusion_top_k
    fused: dict[int, dict[str, Any]] = {}
    for source, results in (("vector", vector_results), ("bm25", bm25_results)):
        for item in results:
            chunk_id = int(item["chunk_id"])
            record = fused.setdefault(chunk_id, {**item, "vector_rank": None, "bm25_rank": None, "rrf_score": 0.0})
            record[f"{source}_rank"] = item["rank"]
            record["rrf_score"] += 1 / (k + item["rank"])
    ranked = sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)[:top_k]
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank; item["rrf_score"] = round(item["rrf_score"], 8)
    return ranked


def hybrid_search(query: str, class_id: int, embedder=None, collection=None) -> dict:
    vectors = vector_search(query, class_id, embedder=embedder, collection=collection)
    keywords = bm25_search(query, class_id)
    return {"vector": vectors, "bm25": keywords, "rrf": rrf_fuse(vectors, keywords)}


def rerank_search(query: str, candidates: list[dict], reranker=None, top_k: int | None = None) -> list[dict]:
    if not candidates:
        return []
    top_k = top_k or settings.rag_final_top_k
    rankings = (reranker or get_rerank_client()).rerank(query, [item["content"] for item in candidates])
    output: list[dict] = []
    for rank, result in enumerate(rankings[:top_k], start=1):
        index = result["index"]
        if index < 0 or index >= len(candidates):
            continue
        output.append({
            **candidates[index],
            "rank": rank,
            "rerank_score": round(float(result["score"]), 6),
        })
    return output


def resolve_parent_chunks(candidates: list[dict], class_id: int) -> list[dict]:
    parent_ids = list(dict.fromkeys(int(item["parent_id"]) for item in candidates if item.get("parent_id")))
    if not parent_ids:
        return []
    with SessionLocal() as db:
        rows = db.execute(
            select(KnowledgeChunk, KnowledgeDocument)
            .join(KnowledgeDocument, KnowledgeDocument.id == KnowledgeChunk.document_id)
            .where(
                KnowledgeChunk.id.in_(parent_ids),
                KnowledgeChunk.chunk_type == "parent",
                KnowledgeDocument.class_id == class_id,
                KnowledgeDocument.enabled.is_(True),
                KnowledgeDocument.index_status == "indexed",
            )
        ).all()
    by_id = {chunk.id: (chunk, document) for chunk, document in rows}
    output: list[dict] = []
    for candidate in candidates:
        parent_id = int(candidate.get("parent_id") or 0)
        if parent_id not in by_id or any(item["parent_id"] == parent_id for item in output):
            continue
        chunk, document = by_id[parent_id]
        output.append({
            "parent_id": chunk.id,
            "document_id": document.id,
            "filename": document.filename,
            "heading": chunk.heading,
            "section_path": chunk.section_path,
            "page": chunk.page,
            "content": chunk.content,
            "rank": len(output) + 1,
            "source_label": f"资料{len(output) + 1}",
            "matched_chunk_id": candidate["chunk_id"],
            "rerank_score": candidate.get("rerank_score", candidate.get("score", 0.0)),
        })
    return output


def retrieve_with_rerank(query: str, class_id: int, embedder=None, collection=None, reranker=None) -> dict:
    recalled = hybrid_search(query, class_id, embedder=embedder, collection=collection)
    reranked = rerank_search(query, recalled["rrf"], reranker=reranker)
    return {**recalled, "rerank": reranked, "parents": resolve_parent_chunks(reranked, class_id)}

