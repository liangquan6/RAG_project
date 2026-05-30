"""
Agent 中间件模块

功能概述：
- 工具调用监控（记录日志）
- 模型调用前日志
- 动态提示词切换（普通对话 vs 报告生成）

中间件类型：
- wrap_tool_call: 包裹工具调用
- before_model: 模型调用前执行
- dynamic_prompt: 动态生成提示词
"""

from typing import Callable
from utils.prompt_loader import load_system_prompts, load_report_prompts
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger


@wrap_tool_call
def monitor_tool(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """
    工具调用监控中间件
    
    功能：
    1. 记录工具调用日志
    2. 捕获工具调用异常
    3. 特殊处理：调用 fill_context_for_report 时设置报告模式标记
    
    参数:
        request: 工具调用请求对象（包含工具名、参数等）
        handler: 实际执行工具的函数
        
    返回:
        工具执行结果
    """
    # 记录工具调用开始
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")

    try:
        # 执行实际的工具调用
        result = handler(request)
        # 记录调用成功
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

        # 特殊逻辑：如果调用的是 fill_context_for_report 工具
        # 则在 runtime context 中设置 report = True
        # 这样后续的 dynamic_prompt 中间件会切换到报告提示词
        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True

        return result
    except Exception as e:
        # 记录工具调用失败
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e


@before_model
def log_before_model(
        state: AgentState,
        runtime: Runtime,
):
    """
    模型调用前日志中间件
    
    功能：
    在每次调用大语言模型前，记录消息数量和最后一条消息内容
    
    参数:
        state: Agent 状态对象（包含所有历史消息）
        runtime: 运行时上下文对象
    """
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")

    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")

    return None


@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    """
    动态提示词切换中间件
    
    功能：
    根据 runtime.context["report"] 的值，动态选择提示词：
    - report = True: 使用报告生成提示词
    - report = False: 使用普通对话提示词
    
    参数:
        request: 模型请求对象
        
    返回:
        系统提示词字符串
    """
    # 从上下文中获取报告模式标记
    is_report = request.runtime.context.get("report", False)
    if is_report:
        # 报告生成场景：返回报告提示词
        return load_report_prompts()

    # 默认场景：返回普通对话提示词
    return load_system_prompts()
