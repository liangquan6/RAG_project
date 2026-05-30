"""
对话记忆管理：滑动窗口截断历史，供 Agent / RAG 使用。
"""

from __future__ import annotations

from typing import Any

from utils.config_handler import agent_config
from utils.logger_handler import logger

DEFAULT_MAX_MESSAGES = 10


def get_max_history_messages() -> int:
    conv = agent_config.get("conversation") or {}
    return int(conv.get("max_messages", DEFAULT_MAX_MESSAGES))


def trim_history(
    messages: list[dict[str, Any]] | None,
    max_messages: int | None = None,
) -> list[dict[str, Any]]:
    """
    保留最近 max_messages 条消息（每条为一轮 user 或 assistant）。

    参数:
        messages: [{"role": "user"|"assistant", "content": "..."}, ...]
        max_messages: 上限，默认读配置

    返回:
        截断后的新列表（不修改原列表）
    """
    if not messages:
        return []

    limit = max_messages if max_messages is not None else get_max_history_messages()
    trimmed = messages[-limit:]
    if len(messages) > limit:
        logger.debug(
            f"[Memory] 历史从 {len(messages)} 条截断为 {len(trimmed)} 条"
        )
    return trimmed


def prepare_agent_messages(
    history: list[dict[str, Any]] | None,
    query: str,
    max_messages: int | None = None,
) -> list[dict[str, str]]:
    """
    组装送入 Agent 的 messages 列表（含当前用户输入）。

    若 history 末尾已是当前 user 消息，则不再重复追加。
    """
    query = query.strip()
    msgs = trim_history(history, max_messages)

    if msgs and msgs[-1].get("role") == "user":
        last_content = (msgs[-1].get("content") or "").strip()
        if last_content == query:
            return [
                {"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in msgs
                if m.get("content")
            ]

    result = [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        for m in msgs
        if m.get("content")
    ]
    if not result or result[-1].get("role") != "user" or result[-1]["content"].strip() != query:
        result.append({"role": "user", "content": query})
    return result


def history_for_rag(
    history: list[dict[str, Any]] | None,
    max_messages: int | None = None,
) -> list[dict[str, Any]]:
    """
    供 RAG Query 改写使用的对话历史（不含当前轮时可由调用方决定）。

    默认与 Agent 使用相同窗口；仅保留 user/assistant。
    """
    trimmed = trim_history(history, max_messages)
    return [
        {"role": m["role"], "content": m["content"]}
        for m in trimmed
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
