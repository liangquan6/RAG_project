"""
模型工厂模块

功能概述：
- 使用工厂模式创建大语言模型和嵌入模型
- 便于后续扩展不同的模型提供商
- 统一模型创建接口

当前实现：
- ChatModelFactory: 创建通义千问聊天模型
- EmbeddingsFactory: 创建虚拟嵌入模型（需替换为真实模型）
"""

from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
# from langchain_community.embeddings import FakeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from utils.config_handler import rag_config

from langchain_community.embeddings import DashScopeEmbeddings

class BaseModelFactory(ABC):
    """
    模型工厂抽象基类
    
    所有具体模型工厂都需要继承此类并实现 generate 方法
    """
    @abstractmethod
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        """
        创建模型实例
        
        返回:
            模型实例（Embeddings 或 BaseChatModel）
        """
        pass


class ChatModelFactory(BaseModelFactory):
    """
    聊天模型工厂
    
    创建通义千问 ChatTongyi 模型
    """
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        """
        创建 ChatTongyi 模型实例
        
        返回:
            ChatTongyi 模型实例
        """
        return ChatTongyi(model_name=rag_config["chat_model_name"])




class EmbeddingsFactory(BaseModelFactory):
    def generate(self) -> Embeddings:
        return DashScopeEmbeddings(
            model=rag_config["embedding_model_name"]
        )

# ==========================================
# 全局单例模型实例
# ==========================================
# 聊天模型单例
chat_model = ChatModelFactory().generate()
# 嵌入模型单例
embed_model = EmbeddingsFactory().generate()