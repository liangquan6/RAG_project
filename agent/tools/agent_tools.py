"""
Agent 工具模块

功能概述：
- 定义 Agent 可用的工具函数
- 包括 RAG 检索、天气查询、用户信息获取等
- 使用 @tool 装饰器注册为 LangChain 工具
"""

import os
from utils.logger_handler import logger
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
from utils.conversation_context import get_chat_history
from utils.weather_api import get_weather_service
from utils.location_api import get_location_service
import random
from utils.config_handler import agent_config
from utils.path_tools import get_abs_path

# ==========================================
# 全局初始化
# ==========================================
# 初始化 RAG 服务实例
rag = RagSummarizeService()

# 用户 ID 池（用于模拟随机选择用户）
user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010",]
# 月份数组（用于模拟随机选择月份）
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", ]

# 外部数据缓存字典（用于存储从 CSV 加载的用户使用记录）
external_data = {}


# ==========================================
# 工具函数定义
# ==========================================

@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    """
    RAG 检索工具：根据查询从知识库中检索相关文档并生成总结
    
    参数:
        query: 用户的查询字符串
        
    返回:
        基于知识库的总结回复
    """
    history = get_chat_history()
    return rag.rag_summarize(query, history=history if history else None)


@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    """
    天气查询工具：获取指定城市的天气信息
    
    支持的 API 提供商（通过 config/agent.yml 配置）：
    - mock: 模拟数据（默认）
    - xinzhi: 心知天气 API
    - hefeng: 和风天气 API
    
    参数:
        city: 城市名称
        
    返回:
        天气信息字符串
    """
    weather_service = get_weather_service()
    return weather_service.get_weather(city)

@tool(description="获取用户所在城市的名称，以纯字符串的形式返回")
def get_user_location() -> str:
    """
    用户位置工具：获取用户当前所在城市
    
    支持的位置提供商（通过 config/agent.yml 配置）：
    - mock: 模拟数据（随机城市）
    - ip_api: IP 地理定位（真实位置）
    
    返回:
        城市名称
    """
    location_service = get_location_service()
    return location_service.get_location()


@tool(description="获取用户的ID，以纯字符串的形式返回")
def get_user_id() -> str:
    """
    用户 ID 工具：获取当前用户的 ID
    
    返回:
        用户 ID（目前是随机从池中选择）
    """
    return random.choice(user_ids)


@tool(description="获取当前月份，以纯字符串的形式返回")
def get_current_month() -> str:
    """
    当前月份工具：获取当前月份
    
    返回:
        月份字符串（格式：YYYY-MM，目前是随机模拟）
    """
    return random.choice(month_arr)


def generate_external_data():
    """
    从 CSV 文件加载外部数据（用户使用记录）
    
    数据结构:
        {
            "user_id": {
                "month": {"特征": xxx, "效率": xxx, "耗材": xxx, "对比": xxx},
                ...
            },
            ...
        }
    
    注意：数据会缓存到 external_data 全局变量，避免重复加载
    """
    # 如果数据未加载过，则进行加载
    if not external_data:
        external_data_path = get_abs_path(agent_config["external_data_path"])

        # 检查文件是否存在
        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        # 逐行读取 CSV（跳过表头）
        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                # 按逗号分割，并去除引号
                arr: list[str] = line.strip().split(",")

                user_id: str = arr[0].replace('"', "")
                feature: str = arr[1].replace('"', "")
                efficiency: str = arr[2].replace('"', "")
                consumables: str = arr[3].replace('"', "")
                comparison: str = arr[4].replace('"', "")
                time: str = arr[5].replace('"', "")

                # 初始化用户字典（如果不存在）
                if user_id not in external_data:
                    external_data[user_id] = {}

                # 存储用户在该月份的数据
                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }


@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串的形式返回， 如果未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    """
    外部数据查询工具：获取指定用户在指定月份的使用记录
    
    参数:
        user_id: 用户 ID
        month: 月份（格式：YYYY-MM）
        
    返回:
        该用户在该月份的使用记录数据，未找到返回空字符串
    """
    # 确保数据已加载
    generate_external_data()

    try:
        # 尝试获取数据
        return external_data[user_id][month]
    except KeyError:
        # 未找到数据，记录警告并返回空字符串
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        return ""


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    """
    报告上下文注入工具：触发中间件设置报告生成模式
    
    作用：
        - 调用后会触发 monitor_tool 中间件
        - 将 runtime.context["report"] 设置为 True
        - 后续会触发 report_prompt_switch 动态切换到报告提示词
    """
    return "fill_context_for_report已调用"


if __name__ == "__main__":
    # 测试代码：加载数据并测试查询
    generate_external_data()
    print(external_data)
    # print(fetch_external_data("1021", "2025-01"))
    print(get_user_location())
