"""
查询改写模块：将用户问题扩展为多条检索友好的 query，用于多路召回。
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from model.factory import chat_model
from utils.config_handler import rag_config
from utils.logger_handler import logger

_REWRITE_SYSTEM = """你是扫地机器人知识库检索助手。
根据用户问题，生成用于向量检索的查询变体。
要求：
1. 只输出 JSON 数组，不要 markdown 或其它说明
2. 数组元素为字符串，每条 8~30 字，表述不同但语义贴近原问题
3. 不要重复原问题原文，不要编号
示例：["尘盒清理频率", "扫地机器人尘盒多久倒一次"]"""


class QueryRewriter:
    """查询改写器：LLM 生成多条检索 query。"""

    def __init__(self, extra_count: int | None = None):
        self.extra_count = extra_count or rag_config.get("rewrite_extra_count", 2)
        self.llm = chat_model
        self.parser = StrOutputParser()

    def _format_history(self, history: list[dict[str, Any]] | None) -> str:
        if not history:
            return ""
        lines = []
        for msg in history[-4:]:
            role = msg.get("role", "user")
            content = (msg.get("content") or "").strip()
            if content:
                lines.append(f"{role}: {content[:200]}")
        if not lines:
            return ""
        return "近期对话：\n" + "\n".join(lines) + "\n\n"

    def _parse_queries(self, raw: str, original: str) -> list[str]:
        raw = raw.strip()
        candidates: list[str] = []

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                candidates = [str(x).strip() for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            pass

        if not candidates:
            for line in raw.splitlines():
                line = re.sub(r"^[\d\-\.\)、\s]+", "", line).strip().strip('"\'')
                if 4 <= len(line) <= 80:
                    candidates.append(line)

        seen = {original.strip()}
        result = [original.strip()]
        for q in candidates:
            if q not in seen:
                seen.add(q)
                result.append(q)
            if len(result) >= 1 + self.extra_count:
                break
        return result

    def rewrite(self, query: str, history: list[dict[str, Any]] | None = None) -> list[str]:
        """
        将用户问题改写成多条检索 query（含原问题）。

        参数:
            query: 用户原始问题
            history: 可选，[{"role":"user"|"assistant","content":"..."}, ...]

        返回:
            去重后的 query 列表，第一项为原问题
        """
        query = query.strip()
        if not query:
            return [""]

        history_text = self._format_history(history)
        user_prompt = (
            f"{history_text}"
            f"用户问题：{query}\n"
            f"请再生成 {self.extra_count} 条不同表述的检索查询，输出 JSON 数组。"
        )

        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=_REWRITE_SYSTEM),
                    HumanMessage(content=user_prompt),
                ]
            )
            raw = self.parser.invoke(response)
            queries = self._parse_queries(raw, query)
            logger.info(f"[QueryRewriter] {query!r} -> {queries}")
            return queries
        except Exception as e:
            logger.warning(f"[QueryRewriter] 改写失败，使用原问题: {e}")
            return [query]


if __name__ == "__main__":
    rewriter = QueryRewriter()
    print(rewriter.rewrite("扫地机器人吸力变弱怎么办"))
