"""
请求级对话上下文（ContextVar），供 Tool 在执行时读取当前会话历史。
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

_chat_history: ContextVar[list[dict[str, Any]]] = ContextVar(
    "chat_history",
    default=[],
)


def set_chat_history(history: list[dict[str, Any]] | None) -> None:
    """在当前请求/流式调用开始时注入对话历史。"""
    _chat_history.set(list(history) if history else [])


def get_chat_history() -> list[dict[str, Any]]:
    """Tool 内获取当前对话历史（可能为空列表）。"""
    return list(_chat_history.get())


def clear_chat_history() -> None:
    _chat_history.set([])
