"""
天气 API 服务模块

支持的 API 提供商：
- mock: 模拟数据（用于测试）
- xinzhi: 心知天气 API
- hefeng: 和风天气 API

免费 API 申请地址：
- 心知天气：https://www.seniverse.com/
- 和风天气：https://dev.qweather.com/
"""

import os
import json
import httpx
from typing import Optional, Dict, Any
from utils.logger_handler import logger
from utils.config_handler import agent_config
from utils.retry_decorator import retry


class WeatherAPIError(Exception):
    """天气 API 异常"""
    pass


@retry(
    max_attempts=3,
    base_delay=1.0,
    backoff_factor=2.0,
    exceptions=(httpx.TimeoutException, httpx.ConnectError, httpx.TransportError)
)
def make_http_request(
    url: str,
    params: Optional[Dict] = None,
    timeout: float = 10.0
) -> Dict:
    """
    发起 HTTP 请求（带重试机制）
    
    参数:
        url: 请求 URL
        params: 请求参数
        timeout: 超时时间（秒）
    
    返回:
        JSON 响应数据
    """
    response = httpx.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


class WeatherService:
    """
    天气服务类
    
    支持多种 API 提供商，通过配置文件灵活切换
    """
    
    def __init__(self):
        self.config = agent_config.get("weather_api", {})
        self.provider = self.config.get("provider", "mock")
        
    def get_weather(self, city: str) -> str:
        """
        获取天气信息
        
        参数:
            city: 城市名称
            
        返回:
            格式化的天气信息字符串
        """
        if self.provider == "mock":
            return self._get_mock_weather(city)
        elif self.provider == "xinzhi":
            return self._get_xinzhi_weather(city)
        elif self.provider == "hefeng":
            return self._get_hefeng_weather(city)
        else:
            logger.warning(f"未知的天气提供商: {self.provider}，使用模拟数据")
            return self._get_mock_weather(city)
    
    def _get_mock_weather(self, city: str) -> str:
        """
        获取模拟天气数据
        
        参数:
            city: 城市名称
            
        返回:
            模拟的天气信息字符串
        """
        return f"城市{city}天气为晴天，气温26摄氏度，空气湿度50%，南风1级，AQI21，最近6小时降雨概率极低"
    
    def _get_xinzhi_weather(self, city: str) -> str:
        """
        获取心知天气 API 数据
        
        心知天气免费 API 限制：
        - 400次/小时
        - 城市名称需使用拼音或城市代码
        
        参数:
            city: 城市名称
            
        返回:
            格式化的天气信息字符串
        """
        api_key = self.config.get("xinzhi_key") or os.getenv("WEATHER_API_KEY")
        api_url = self.config.get("xinzhi_url")
        
        if not api_key:
            logger.warning("未配置心知天气 API Key，使用模拟数据")
            return self._get_mock_weather(city)
        
        try:
            # 心知天气 API 城市参数需用拼音
            city_pinyin = self._city_to_pinyin(city)
            
            params = {
                "key": api_key,
                "location": city_pinyin,
                "language": "zh-Hans",
                "unit": "c"
            }
            
            # 带重试的 HTTP 请求
            data = make_http_request(api_url, params=params, timeout=10.0)
            
            # 检查 API 返回状态
            # 心知天气成功时返回 results 字段，失败时返回 status 错误信息
            if "results" not in data or not data["results"]:
                logger.warning(f"心知天气 API 返回错误: {data}")
                return self._get_mock_weather(city)
            
            # 解析天气数据
            weather_data = data["results"][0]["now"]
            
            return self._format_xinzhi_weather(city, weather_data)
            
        except Exception as e:
            logger.error(f"调用心知天气 API 失败: {str(e)}")
            return self._get_mock_weather(city)
    
    def _get_hefeng_weather(self, city: str) -> str:
        """
        获取和风天气 API 数据
        
        和风天气免费 API 限制：
        - 1000次/天
        - 需要先查询城市代码
        
        参数:
            city: 城市名称
            
        返回:
            格式化的天气信息字符串
        """
        api_key = self.config.get("hefeng_key") or os.getenv("HEFENG_WEATHER_API_KEY")
        api_url = self.config.get("hefeng_url")
        
        if not api_key:
            logger.warning("未配置和风天气 API Key，使用模拟数据")
            return self._get_mock_weather(city)
        
        try:
            # 先查询城市代码
            location_url = "https://geoapi.qweather.com/v2/city/lookup"
            location_params = {
                "key": api_key,
                "location": city,
                "lang": "zh"
            }
            
            location_response = httpx.get(location_url, params=location_params, timeout=10)
            location_data = location_response.json()
            
            if location_data.get("code") != "200":
                logger.warning(f"和风天气城市查询失败: {location_data}")
                return self._get_mock_weather(city)
            
            city_id = location_data["location"][0]["id"]
            
            # 获取天气数据
            weather_params = {
                "key": api_key,
                "location": city_id,
                "lang": "zh"
            }
            
            weather_response = httpx.get(api_url, params=weather_params, timeout=10)
            weather_data = weather_response.json()
            
            if weather_data.get("code") != "200":
                logger.warning(f"和风天气 API 返回错误: {weather_data}")
                return self._get_mock_weather(city)
            
            return self._format_hefeng_weather(city, weather_data["now"])
            
        except Exception as e:
            logger.error(f"调用和风天气 API 失败: {str(e)}")
            return self._get_mock_weather(city)
    
    def _city_to_pinyin(self, city: str) -> str:
        """
        城市名称转拼音（简化版）
        
        注意：这是一个简化实现，仅支持部分常用城市
        生产环境建议使用完整的中文拼音转换库
        
        参数:
            city: 城市名称
            
        返回:
            城市拼音
        """
        city_map = {
            "北京": "beijing",
            "上海": "shanghai",
            "广州": "guangzhou",
            "深圳": "shenzhen",
            "杭州": "hangzhou",
            "成都": "chengdu",
            "武汉": "wuhan",
            "西安": "xian",
            "南京": "nanjing",
            "重庆": "chongqing",
            "天津": "tianjin",
            "苏州": "suzhou",
            "郑州": "zhengzhou",
            "长沙": "changsha",
            "沈阳": "shenyang",
            "青岛": "qingdao",
            "宁波": "ningbo",
            "东莞": "dongguan",
            "合肥": "hefei",
            "昆明": "kunming",
        }
        return city_map.get(city, city)
    
    def _format_xinzhi_weather(self, city: str, data: Dict[str, Any]) -> str:
        """
        格式化心知天气数据
        
        参数:
            city: 城市名称
            data: 心知天气返回的天气数据
            
        返回:
            格式化的天气信息字符串
        """
        # 免费版 API 只返回基本字段，需要处理缺失的字段
        text = data.get('text', '未知')
        temperature = data.get('temperature', '未知')
        feels_like = data.get('feels_like', '未知')
        humidity = data.get('humidity', '未知')
        wind_direction = data.get('wind_direction', '')
        wind_scale = data.get('wind_scale', '')
        
        # 构建天气字符串
        result = f"城市{city}天气{text}"
        
        if temperature != '未知':
            result += f"，气温{temperature}摄氏度"
        
        if feels_like != '未知':
            result += f"，体感温度{feels_like}摄氏度"
        
        if humidity != '未知':
            result += f"，空气湿度{humidity}%"
        
        if wind_direction or wind_scale:
            result += f"，{wind_direction}{wind_scale}级"
        
        # 添加 AQI（如果有）
        air_quality = data.get('air_quality', {})
        if air_quality:
            aqi = air_quality.get('aqi', '未知')
            result += f"，AQI{aqi}"
        
        return result
    
    def _format_hefeng_weather(self, city: str, data: Dict[str, Any]) -> str:
        """
        格式化和风天气数据
        
        参数:
            city: 城市名称
            data: 和风天气返回的天气数据
            
        返回:
            格式化的天气信息字符串
        """
        return (
            f"城市{city}天气{data['text']}，"
            f"气温{data['temp']}摄氏度，"
            f"体感温度{data['feelsLike']}摄氏度，"
            f"空气湿度{data['humidity']}%，"
            f"{data['windDir']}{data['windScale']}级，"
            f"能见度{data['vis']}公里，"
            f"体感舒适度{data.get('comfort', '未知')}"
        )


# 全局单例
_weather_service: Optional[WeatherService] = None


def get_weather_service() -> WeatherService:
    """
    获取天气服务单例
    
    返回:
        WeatherService 实例
    """
    global _weather_service
    if _weather_service is None:
        _weather_service = WeatherService()
    return _weather_service
