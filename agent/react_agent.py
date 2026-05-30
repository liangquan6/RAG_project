"""
ReAct Agent 智能体模块

功能概述：
- 基于 LangChain 创建 ReAct 推理智能体
- 集成多个工具（RAG 检索、天气、用户信息等）
- 支持中间件（监控、日志、动态提示词切换）
- 支持多轮对话记忆（滑动窗口）
- 提供流式输出接口
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent

from agent.tools.agent_tools import (
    fetch_external_data,
    fill_context_for_report,
    get_current_month,
    get_user_id,
    get_user_location,
    get_weather,
    rag_summarize,
)
from agent.tools.middleware import log_before_model, monitor_tool, report_prompt_switch
from model.factory import chat_model
from utils.conversation_context import clear_chat_history, set_chat_history
from utils.conversation_memory import history_for_rag, prepare_agent_messages
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts


class ReactAgent:
    """
    ReAct 智能体类

    ReAct = Reasoning + Acting（推理 + 行动）
    智能体通过「思考-行动-观察」循环完成任务。
    """

    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[
                rag_summarize,
                get_weather,
                get_user_location,
                get_user_id,
                get_current_month,
                fetch_external_data,
                fill_context_for_report,
            ],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    def execute_stream(
        self,
        query: str,
        history: list[dict[str, Any]] | None = None,
    ):
        """
        执行流式对话（支持多轮历史）。

        参数:
            query: 当前用户输入
            history: 可选，[{"role":"user"|"assistant","content":"..."}, ...]
                       通常包含本轮 user 消息（与 app 的 session_state.messages 一致）

        返回:
            流式文本块生成器
        """
        messages = prepare_agent_messages(history, query)
        rag_history = history_for_rag(history)

        logger.info(
            f"[Agent] 多轮上下文: agent_msgs={len(messages)}, rag_history={len(rag_history)}"
        )

        set_chat_history(rag_history)
        input_dict = {"messages": messages}

        try:
            for chunk in self.agent.stream(
                input_dict,
                stream_mode="values",
                context={"report": False},
            ):
                latest_message = chunk["messages"][-1]
                if latest_message.content:
                    yield latest_message.content.strip() + "\n"
        finally:
            clear_chat_history()


if __name__ == "__main__":
    agent = ReactAgent()
    hist = [
        {"role": "user", "content": "长沙今天天气怎么样"},
        {"role": "assistant", "content": "长沙今天阴，25度。"},
    ]
    for chunk in agent.execute_stream("那机器人该怎么保养", history=hist):
        print(chunk, end="", flush=True)
