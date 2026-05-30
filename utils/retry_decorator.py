"""
重试装饰器模块

功能概述：
- 为网络请求、API 调用提供自动重试机制
- 支持指数退避策略
- 记录重试日志
"""

import time
import random
from functools import wraps
from typing import Callable, Type, Tuple, Union
from utils.logger_handler import logger


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Callable[[int, Exception], None] = None,
) -> Callable:
    """
    重试装饰器
    
    参数:
        max_attempts: 最大重试次数
        base_delay: 初始延迟（秒）
        max_delay: 最大延迟（秒）
        backoff_factor: 退避因子（每次延迟 = 上次延迟 * backoff_factor）
        jitter: 是否添加随机抖动（避免多个请求同时重试）
        exceptions: 需要重试的异常类型
        on_retry: 重试回调函数，参数为 (重试次数, 异常对象)
    
    使用示例:
        @retry(max_attempts=3, exceptions=(ConnectionError, TimeoutError))
        def fetch_data(url):
            # ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            delay = base_delay
            
            while attempt < max_attempts:
                attempt += 1
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt >= max_attempts:
                        logger.error(f"函数 {func.__name__} 已达到最大重试次数 {max_attempts}，放弃重试")
                        raise
                    
                    # 计算下次延迟（指数退避）
                    delay = min(delay * backoff_factor, max_delay)
                    
                    # 添加随机抖动
                    if jitter:
                        delay = delay * (0.5 + random.random())
                    
                    if on_retry:
                        on_retry(attempt, e)
                    
                    logger.warning(
                        f"函数 {func.__name__} 执行失败（尝试 {attempt}/{max_attempts}），"
                        f"{delay:.2f} 秒后重试: {str(e)}"
                    )
                    
                    time.sleep(delay)
            
            # 理论上不会到达这里
            raise RuntimeError(f"函数 {func.__name__} 重试失败")
        
        return wrapper
    return decorator


def async_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Callable[[int, Exception], None] = None,
) -> Callable:
    """
    异步函数重试装饰器
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            import asyncio
            attempt = 0
            delay = base_delay
            
            while attempt < max_attempts:
                attempt += 1
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt >= max_attempts:
                        logger.error(f"函数 {func.__name__} 已达到最大重试次数 {max_attempts}，放弃重试")
                        raise
                    
                    delay = min(delay * backoff_factor, max_delay)
                    
                    if jitter:
                        delay = delay * (0.5 + random.random())
                    
                    if on_retry:
                        on_retry(attempt, e)
                    
                    logger.warning(
                        f"函数 {func.__name__} 执行失败（尝试 {attempt}/{max_attempts}），"
                        f"{delay:.2f} 秒后重试: {str(e)}"
                    )
                    
                    await asyncio.sleep(delay)
            
            raise RuntimeError(f"函数 {func.__name__} 重试失败")
        
        return wrapper
    return decorator


def simple_retry(max_attempts: int = 3, delay: float = 1.0) -> Callable:
    """
    简单重试装饰器（固定延迟）
    
    参数:
        max_attempts: 最大重试次数
        delay: 每次重试之间的延迟（秒）
    """
    return retry(
        max_attempts=max_attempts,
        base_delay=delay,
        max_delay=delay,
        backoff_factor=1.0,
        jitter=False
    )
