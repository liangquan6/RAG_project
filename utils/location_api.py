"""
用户位置服务模块

支持的位置获取方式：
- mock: 模拟数据（随机城市）
- ip_api: 使用 ip-api.com 免费 API（不需要 API Key）
"""

import os
import httpx
from typing import Optional, Dict, Any
from utils.logger_handler import logger
from utils.config_handler import agent_config
from utils.retry_decorator import retry
import random


class LocationAPIError(Exception):
    """位置 API 异常"""
    pass


@retry(
    max_attempts=3,
    base_delay=1.0,
    backoff_factor=2.0,
    exceptions=(httpx.TimeoutException, httpx.ConnectError)
)
def get_location_by_ip(ip: Optional[str] = None) -> Dict[str, Any]:
    """
    通过 IP 地址获取地理位置（使用 ip-api.com 免费 API）
    
    参数:
        ip: 可选，指定 IP 地址，不指定则使用当前请求 IP
    
    返回:
        地理位置信息字典
    """
    url = "http://ip-api.com/json/" + (ip or "")
    response = httpx.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


class LocationService:
    """
    用户位置服务类
    
    支持多种位置获取方式，通过配置文件切换
    """
    
    def __init__(self):
        # 每次初始化都重新加载配置
        from utils.config_handler import load_agent_config
        self.config = load_agent_config().get("location_api", {})
        self.provider = self.config.get("provider", "mock")
        
        # 模拟城市列表
        self.mock_cities = [
            "北京", "上海", "广州", "深圳", "杭州", "成都",
            "武汉", "南京", "西安", "重庆", "天津", "苏州",
            "郑州", "长沙", "沈阳", "青岛", "宁波", "东莞",
            "合肥", "昆明"
        ]
    
    def get_location(self, ip: Optional[str] = None) -> str:
        """
        获取用户位置（城市名称）
        
        参数:
            ip: 可选，指定 IP 地址
        
        返回:
            城市名称
        """
        # 每次调用都重新读取配置，支持动态切换
        from utils.config_handler import load_agent_config
        current_provider = load_agent_config().get("location_api", {}).get("provider", "mock")
        
        if current_provider == "mock":
            return self._get_mock_location()
        elif current_provider == "ip_api":
            return self._get_ip_location(ip)
        else:
            logger.warning(f"未知的位置提供商: {current_provider}，使用模拟数据")
            return self._get_mock_location()
    
    def _get_mock_location(self) -> str:
        """
        获取模拟位置
        
        返回:
            随机城市名称
        """
        return random.choice(self.mock_cities)
    
    def _get_ip_location(self, ip: Optional[str] = None) -> str:
        """
        通过 IP 获取真实位置
        
        参数:
            ip: 可选，指定 IP 地址
        
        返回:
            城市名称
        """
        try:
            data = get_location_by_ip(ip)
            
            # 检查 API 返回状态
            if data.get("status") != "success":
                logger.warning(f"IP 地理定位 API 返回错误: {data.get('message', '未知错误')}")
                return self._get_mock_location()
            
            # 返回城市名称（优先使用中文城市名）
            city = data.get("city")
            if not city:
                city = data.get("regionName")
            
            if not city:
                logger.warning("无法获取城市信息")
                return self._get_mock_location()
            
            return city
            
        except Exception as e:
            logger.error(f"调用 IP 地理定位 API 失败: {str(e)}")
            return self._get_mock_location()


# 全局单例
_location_service: Optional[LocationService] = None


def get_location_service() -> LocationService:
    """
    获取位置服务单例
    
    返回:
        LocationService 实例
    """
    global _location_service
    if _location_service is None:
        _location_service = LocationService()
    return _location_service
