"""
RAG 评估执行器

流程：
1. 加载评测集
2. 检索评估（Hit@K / MRR@K）
3. 可选：Ragas 生成评估（Faithfulness / Answer Relevancy / Context Precision）
4. 输出 JSON 报告
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.config_handler import chroma_config
from utils.logger_handler import logger
from utils.path_tools import get_abs_path

from eval.metrics.retrieval import hit_at_k, mrr_at_k


DEFAULT_DATASET = get_abs_path("eval/datasets/rag_eval.json")
DEFAULT_RESULTS_DIR = get_abs_path("eval/results")


class RAGEvaluator:
    def __init__(self, with_ragas: bool = False, similarity_threshold: float = 0.25):
        self.with_ragas = with_ragas
        self.similarity_threshold = similarity_threshold
        self.k = chroma_config["k"]
        self._rag_service = None

    @property
    def rag_service(self):
        if self._rag_service is None:
            from rag.rag_service import RagSummarizeService

            logger.info("[RAG Eval] 初始化 RAG 服务（含知识库检查）...")
            self._rag_service = RagSummarizeService(verbose=False)
        return self._rag_service

    @staticmethod
    def load_dataset(dataset_path: str) -> list[dict[str, Any]]:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"评测集不存在: {dataset_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list) or not data:
            raise ValueError("评测集格式错误，应为非空 JSON 数组")

        for item in data:
            if not item.get("question") or not item.get("reference_contexts"):
                raise ValueError(f"样本 {item.get('id', 'unknown')} 缺少 question 或 reference_contexts")

        return data

    def evaluate_retrieval(self, dataset: list[dict[str, Any]]) -> dict[str, Any]:
        hits = 0
        mrr_sum = 0.0
        details: list[dict[str, Any]] = []
        total_latency_ms = 0.0

        for item in dataset:
            question = item["question"]
            references = item["reference_contexts"]

            start = time.perf_counter()
            docs = self.rag_service.retriever_docs(question)
            latency_ms = (time.perf_counter() - start) * 1000
            total_latency_ms += latency_ms

            retrieved = [doc.page_content for doc in docs]
            hit = hit_at_k(retrieved, references, self.k, self.similarity_threshold)
            mrr = mrr_at_k(retrieved, references, self.k, self.similarity_threshold)

            hits += int(hit)
            mrr_sum += mrr

            details.append(
                {
                    "id": item.get("id"),
                    "question": question,
                    "hit": hit,
                    "mrr": round(mrr, 4),
                    "latency_ms": round(latency_ms, 2),
                    "retrieved_count": len(retrieved),
                    "top1_preview": retrieved[0][:120] if retrieved else "",
                }
            )

        total = len(dataset)
        return {
            "k": self.k,
            "similarity_threshold": self.similarity_threshold,
            "total": total,
            "hit_count": hits,
            "hit_rate": round(hits / total, 4) if total else 0.0,
            "mrr": round(mrr_sum / total, 4) if total else 0.0,
            "avg_retrieval_latency_ms": round(total_latency_ms / total, 2) if total else 0.0,
            "details": details,
        }

    def evaluate_generation(self, dataset: list[dict[str, Any]]) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []

        for item in dataset:
            question = item["question"]
            docs = self.rag_service.retriever_docs(question)
            contexts = [doc.page_content for doc in docs]

            start = time.perf_counter()
            answer = self.rag_service.rag_summarize(question)
            latency_ms = (time.perf_counter() - start) * 1000

            rows.append(
                {
                    "question": question,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": item.get("ground_truth", ""),
                }
            )
            logger.info(f"[RAG Eval] 生成完成: {item.get('id')} ({latency_ms:.0f}ms)")

        return self._run_ragas(rows)

    @staticmethod
    def _run_ragas(rows: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.llms import LangchainLLMWrapper
            from ragas.metrics import answer_relevancy, context_precision, faithfulness

            from model.factory import chat_model, embed_model
        except ImportError as e:
            return {"enabled": False, "error": f"Ragas 依赖未安装: {e}"}

        dataset = Dataset.from_list(rows)
        llm = LangchainLLMWrapper(chat_model)
        embeddings = LangchainEmbeddingsWrapper(embed_model)

        try:
            result = evaluate(
                dataset,
                metrics=[context_precision, faithfulness, answer_relevancy],
                llm=llm,
                embeddings=embeddings,
            )
            # ragas 0.4+: result.scores 是 list[dict], 每条样本一个分数 dict, 需按指标求平均
            raw_scores = result.scores if hasattr(result, "scores") else result
            agg: dict[str, list[float]] = {}
            if isinstance(raw_scores, list):
                for row in raw_scores:
                    for k, v in row.items():
                        agg.setdefault(k, []).append(float(v))
                scores = {k: round(sum(v) / len(v), 4) for k, v in agg.items()}
            elif hasattr(raw_scores, "items"):
                scores = {k: round(float(v), 4) for k, v in raw_scores.items()}
            else:
                scores = {}
            return {"enabled": True, "scores": scores, "sample_count": len(rows)}
        except Exception as e:
            logger.error(f"[RAG Eval] Ragas 评估失败: {e}", exc_info=True)
            return {"enabled": True, "error": str(e), "sample_count": len(rows)}

    def run(
        self,
        dataset_path: str = DEFAULT_DATASET,
        output_dir: str = DEFAULT_RESULTS_DIR,
    ) -> dict[str, Any]:
        dataset = self.load_dataset(dataset_path)

        report: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": dataset_path,
            "sample_count": len(dataset),
            "config": {
                "k": self.k,
                "chunk_size": chroma_config.get("chunk_size"),
                "chunk_overlap": chroma_config.get("chunk_overlap"),
            },
        }

        logger.info(f"[RAG Eval] 开始检索评估，样本数={len(dataset)}")
        report["retrieval"] = self.evaluate_retrieval(dataset)

        if self.with_ragas:
            logger.info("[RAG Eval] 开始 Ragas 生成评估（会调用 LLM，耗时较长）")
            report["generation"] = self.evaluate_generation(dataset)
        else:
            report["generation"] = {"enabled": False, "note": "使用 --with-ragas 开启生成评估"}

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_file = Path(output_dir) / f"rag_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        report["output_file"] = str(output_file)
        return report
