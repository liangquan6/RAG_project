"""
RAG 总结服务模块

功能概述：
- 接收用户查询，从向量数据库中检索相关文档
- 可选：Query 改写多路召回 + Embedding 重排序
- 将查询和检索到的参考资料组合成提示词
- 提交给大语言模型，生成基于知识库的智能回复
"""

from __future__ import annotations

import hashlib
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from model.factory import chat_model
from rag.rag_query_rewriter import QueryRewriter
from rag.rag_reranker import RerankService
from rag.vector_store import VectorStoreService
from utils.config_handler import chroma_config, rag_config
from utils.logger_handler import logger
from utils.prompt_loader import load_rag_prompts


def _build_prompt_debugger(verbose: bool):
    def _debug_prompt(prompt):
        if verbose:
            print("-" * 20)
            print(prompt.to_string())
            print("-" * 20)
        return prompt

    return RunnableLambda(_debug_prompt)


def _doc_dedupe_key(doc: Document) -> str:
    text = (doc.page_content or "").strip()
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class RagSummarizeService:
    """
    RAG 总结服务类

    核心流程:
        1. 用户提问 -> 2.（可选）Query 改写 -> 3. 向量检索 -> 4.（可选）Rerank
        -> 5. 拼接上下文 -> 6. 模型回复
    """

    def __init__(
        self,
        verbose: bool = False,
        auto_load: bool = True,
        enable_query_rewrite: bool | None = None,
        enable_rerank: bool | None = None,
    ):
        self.verbose = verbose
        self.enable_query_rewrite = (
            enable_query_rewrite
            if enable_query_rewrite is not None
            else rag_config.get("enable_query_rewrite", False)
        )
        self.enable_rerank = (
            enable_rerank
            if enable_rerank is not None
            else rag_config.get("enable_rerank", False)
        )
        self.final_k = chroma_config["k"]
        self.retrieve_k = rag_config.get("retrieve_k", max(self.final_k * 5, 15))

        self.vector_store = VectorStoreService(auto_load=auto_load)
        self.coarse_retriever = self.vector_store.get_retriever(k=self.retrieve_k)
        self.query_rewriter = QueryRewriter() if self.enable_query_rewrite else None
        self.reranker = RerankService() if self.enable_rerank else None

        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain(verbose)

        logger.info(
            f"[RAG] 初始化完成 rewrite={self.enable_query_rewrite} "
            f"rerank={self.enable_rerank} retrieve_k={self.retrieve_k} final_k={self.final_k}"
        )

    def _init_chain(self, verbose: bool):
        return self.prompt_template | _build_prompt_debugger(verbose) | self.model | StrOutputParser()

    def _merge_documents(self, doc_lists: list[list[Document]]) -> list[Document]:
        seen: set[str] = set()
        merged: list[Document] = []
        for docs in doc_lists:
            for doc in docs:
                key = _doc_dedupe_key(doc)
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)
        return merged

    def retriever_docs(
        self,
        query: str,
        history: list[dict[str, Any]] | None = None,
    ) -> list[Document]:
        """
        检索相关文档（含可选改写 + 多路召回 + 重排序）。

        参数:
            query: 用户查询
            history: 可选对话历史，供 query 改写使用

        返回:
            Document 列表
        """
        queries = [query]
        if self.query_rewriter:
            queries = self.query_rewriter.rewrite(query, history=history)

        all_docs: list[list[Document]] = []
        for q in queries:
            try:
                docs = self.coarse_retriever.invoke(q)
                all_docs.append(docs)
            except Exception as e:
                logger.warning(f"[RAG] 检索失败 query={q!r}: {e}")

        merged = self._merge_documents(all_docs)
        if not merged:
            return []

        if self.reranker:
            return self.reranker.rerank(query, merged, top_k=self.final_k)

        return merged[: self.final_k]

    def rag_summarize(
        self,
        query: str,
        history: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        RAG 主流程：检索 + 总结

        参数:
            query: 用户的问题
            history: 可选对话历史

        返回:
            基于知识库的模型回复
        """
        context_docs = self.retriever_docs(query, history=history)

        context = ""
        for counter, doc in enumerate(context_docs, start=1):
            context += (
                f"【参考资料{counter}】: 参考资料: {doc.page_content} "
                f"| 参考元数据: {doc.metadata}\n"
            )

        return self.chain.invoke({"input": query, "context": context})


if __name__ == "__main__":
    rag_service = RagSummarizeService()
    print(rag_service.rag_summarize("尘盒多久清理一次"))
