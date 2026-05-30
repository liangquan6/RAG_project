"""
yaml配置文件读取
"""

import yaml
import os
from dotenv import load_dotenv
from utils.path_tools import get_abs_path

# 加载 .env 文件
env_path = get_abs_path(".env")
if os.path.exists(env_path):
    load_dotenv(env_path)


def load_rag_config(config_path: str=get_abs_path("config/rag.yml"),encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.safe_load(f)


def load_chroma_config(config_path: str=get_abs_path("config/chroma.yml"),encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.safe_load(f)


def load_prompts_config(config_path: str=get_abs_path("config/prompts.yml"),encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.safe_load(f)


def load_agent_config(config_path: str=get_abs_path("config/agent.yml"),encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.safe_load(f)


rag_config = load_rag_config()
chroma_config = load_chroma_config()
prompts_config = load_prompts_config()
agent_config = load_agent_config()


if __name__ == "__main__":
    print(rag_config["chat_model_name"])


