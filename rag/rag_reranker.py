
"""
重排序模块：对粗排召回的文档按 query 相关性重新排序（embedding 余弦相似度）。
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

from langchain_core.documents import Document
import numpy as np

from model.factory import embed_model
from utils.config_handler import chroma_config
from utils.logger_handler import logger


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _cosine_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """使用 NumPy 优化的余弦相似度计算。"""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


class RerankService:
    """基于 Embedding 的相关性重排序。"""

    def __init__(self):
        self.embeddings = embed_model
        self.default_top_k = chroma_config["k"]

    def rerank(
            self,
            query: str,
            documents: Sequence[Document],
            top_k: int | None = None,
    ) -> list[Document]:
        """
        按与 query 的 embedding 相似度对文档重排并取 Top-K。

        参数:
            query: 用户问题（用原问题排序，不用改写子 query）
            documents: 待排序文档列表
            top_k: 返回条数，默认 chroma.yml 中的 k

        返回:
            重排序后的文档列表
        """
        top_k = top_k or self.default_top_k
        docs = list(documents)
        if not docs:
            return []
        if len(docs) < top_k:
            return docs

        try:
            query_vec = self.embeddings.embed_query(query)
            texts = [doc.page_content for doc in docs]
            doc_vecs = self.embeddings.embed_documents(texts)

            scored = [
                (_cosine_numpy(np.array(query_vec), np.array(vec)), doc)
                for vec, doc in zip(doc_vecs, docs)
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            result = [doc for _, doc in scored[:top_k]]
            logger.info(
                f"[Rerank] {len(docs)} -> top {top_k}, "
                f"best_score={scored[0][0]:.4f}"
            )
            return result
        except Exception as e:
            logger.warning(f"[Rerank] 重排失败，返回原顺序前 {top_k} 条: {e}")
            return docs[:top_k]

    def rerank_with_scores(
            self,
            query: str,
            documents: Sequence[Document],
            top_k: int | None = None,
    ) -> list[Tuple[Document, float]]:
        """
        重排序并返回带分数的结果。

        参数:
            query: 用户问题
            documents: 待排序文档列表
            top_k: 返回条数

        返回:
            (文档, 相似度分数) 的列表
        """
        top_k = top_k or self.default_top_k
        docs = list(documents)
        if not docs:
            return []

        try:
            query_vec = np.array(self.embeddings.embed_query(query))
            texts = [doc.page_content for doc in docs]
            doc_vecs = self.embeddings.embed_documents(texts)

            scored = [
                (_cosine_numpy(query_vec, np.array(vec)), doc)
                for vec, doc in zip(doc_vecs, docs)
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            result = [(doc, score) for score, doc in scored[:top_k]]
            logger.info(
                f"[Rerank] {len(docs)} -> top {top_k}, "
                f"best_score={result[0][1]:.4f}"
            )
            return result
        except Exception as e:
            logger.error(f"[Rerank] 重排失败: {e}")
            raise


if __name__ == "__main__":
    from rag.vector_store import VectorStoreService

    vs = VectorStoreService(auto_load=False)
    retriever = vs.get_retriever(k=10)
    docs = retriever.invoke("尘盒多久清理")
    reranker = RerankService()
    ranked = reranker.rerank("尘盒多久清理", docs, top_k=3)
    for i, d in enumerate(ranked, 1):
        print(i, d.page_content[:80])
