"""
RAG 评估入口

用法:
    python -m eval.run_rag_eval
    python -m eval.run_rag_eval --with-ragas
    python -m eval.run_rag_eval --dataset eval/datasets/rag_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 保证从项目根目录运行时能正确 import
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.runner import DEFAULT_DATASET, RAGEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG 检索与生成评估")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="评测集 JSON 路径",
    )
    parser.add_argument(
        "--with-ragas",
        action="store_true",
        help="开启 Ragas 生成评估（会额外调用 LLM，较慢且消耗 API）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="检索命中相似度阈值（默认 0.5）",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "eval" / "results"),
        help="评估结果输出目录",
    )
    return parser.parse_args()


def print_summary(report: dict) -> None:
    retrieval = report["retrieval"]
    print("\n" + "=" * 60)
    print("RAG 评估结果")
    print("=" * 60)
    print(f"样本数       : {report['sample_count']}")
    print(f"Hit@{retrieval['k']}       : {retrieval['hit_rate']:.2%} ({retrieval['hit_count']}/{retrieval['total']})")
    print(f"MRR@{retrieval['k']}       : {retrieval['mrr']:.4f}")
    print(f"平均检索耗时 : {retrieval['avg_retrieval_latency_ms']:.2f} ms")

    generation = report.get("generation", {})
    if generation.get("enabled") and generation.get("scores"):
        print("\nRagas 生成指标:")
        for name, score in generation["scores"].items():
            print(f"  - {name}: {score:.4f}")
    elif generation.get("enabled") and generation.get("error"):
        print(f"\nRagas 评估失败: {generation['error']}")
    else:
        print("\nRagas 生成评估: 未开启（可加 --with-ragas）")

    print(f"\n报告已保存: {report['output_file']}")
    print("=" * 60)


def main() -> None:
    args = parse_args()
    evaluator = RAGEvaluator(with_ragas=args.with_ragas, similarity_threshold=args.threshold)
    report = evaluator.run(dataset_path=args.dataset, output_dir=args.output_dir)
    print_summary(report)


if __name__ == "__main__":
    main()
