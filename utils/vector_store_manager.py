"""
向量库管理工具

功能概述：
- 检测向量库与当前配置的一致性
- 自动重建向量库（当嵌入模型、分片配置变更时）
- 安全地删除旧向量库数据
"""

import hashlib
import json
import os
import shutil
from datetime import datetime

from model.factory import embed_model
from utils.config_handler import chroma_config, rag_config
from utils.logger_handler import logger
from utils.path_tools import get_abs_path

# 配置文件路径
CONFIG_METADATA_PATH = get_abs_path("chroma_config_metadata.json")


def get_current_config_snapshot() -> dict:
    """
    当前会影响向量写入结果的配置快照（用于哈希对比与排查）
    """
    return {
        "embedding_model": rag_config.get("embedding_model_name", ""),
        "embedding_impl": f"{type(embed_model).__module__}.{type(embed_model).__name__}",
        "chunk_size": chroma_config.get("chunk_size", 0),
        "chunk_overlap": chroma_config.get("chunk_overlap", 0),
        "collection_name": chroma_config.get("collection_name", ""),
        "separators": chroma_config.get("separators", []),
    }


def get_current_config_hash() -> str:
    """
    计算当前配置的稳定哈希值（跨进程一致）
    """
    payload = json.dumps(get_current_config_snapshot(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config_metadata() -> dict:
    """
    加载上次使用的配置元数据

    返回:
        配置元数据字典
    """
    if not os.path.exists(CONFIG_METADATA_PATH):
        return {}

    try:
        with open(CONFIG_METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"加载配置元数据失败: {e}")
        return {}


def save_config_metadata() -> None:
    """
    保存当前配置元数据（入库完成后调用）
    """
    snapshot = get_current_config_snapshot()
    metadata = {
        "config_hash": get_current_config_hash(),
        "config_snapshot": snapshot,
        "last_updated": datetime.now().isoformat(timespec="seconds"),
    }

    try:
        with open(CONFIG_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"保存配置元数据失败: {e}")


def check_vector_store_integrity() -> tuple[bool, str]:
    """
    检查向量库完整性

    返回:
        (是否完整, 原因说明)
    """
    persist_dir = get_abs_path(chroma_config["persist_directory"])

    if not os.path.exists(persist_dir):
        return False, "向量库目录不存在"

    if not os.listdir(persist_dir):
        return False, "向量库目录为空"

    saved_metadata = load_config_metadata()
    saved_hash = saved_metadata.get("config_hash")
    current_hash = get_current_config_hash()

    if not saved_hash:
        return False, "缺少版本元数据，无法确认与当前配置一致"

    if saved_hash != current_hash:
        old = saved_metadata.get("config_snapshot", {})
        new = get_current_config_snapshot()
        return False, (
            f"配置已变更（embedding: {old.get('embedding_model', '未知')} -> {new['embedding_model']}, "
            f"impl: {old.get('embedding_impl', '未知')} -> {new['embedding_impl']}）"
        )

    return True, "向量库完整"


def delete_vector_store(silent: bool = False) -> bool:
    """
    删除向量库及相关增量记录

    参数:
        silent: 是否静默删除（不打印日志）

    返回:
        是否成功删除
    """
    persist_dir = get_abs_path(chroma_config["persist_directory"])
    md5_file = get_abs_path(chroma_config["md5_hex_store"])

    success = True

    if os.path.exists(persist_dir):
        try:
            shutil.rmtree(persist_dir)
            if not silent:
                logger.info(f"已删除向量库: {persist_dir}")
        except Exception as e:
            if not silent:
                logger.error(f"删除向量库失败: {e}")
            success = False

    if os.path.exists(md5_file):
        try:
            os.remove(md5_file)
            if not silent:
                logger.info(f"已删除 MD5 记录: {md5_file}")
        except Exception as e:
            if not silent:
                logger.error(f"删除 MD5 记录失败: {e}")
            success = False

    if os.path.exists(CONFIG_METADATA_PATH):
        try:
            os.remove(CONFIG_METADATA_PATH)
        except Exception as e:
            if not silent:
                logger.warning(f"删除配置元数据失败: {e}")

    return success


def check_and_rebuild_if_needed(force: bool = False) -> bool:
    """
    检查并在需要时清理旧向量库（删除后需重新 load_document）

    参数:
        force: 是否强制重建

    返回:
        True 表示已清理旧库，需要重新入库；False 表示无需清理
    """
    if force:
        logger.info("强制重建向量库...")
        delete_vector_store()
        return True

    is_valid, reason = check_vector_store_integrity()

    if not is_valid:
        logger.info(f"向量库不完整，需要重建: {reason}")
        delete_vector_store()
        return True

    logger.info("向量库完整，无需重建")
    return False


if __name__ == "__main__":
    print("=" * 60)
    print("向量库管理工具")
    print("=" * 60)

    is_valid, reason = check_vector_store_integrity()
    print(f"\n状态: {'完整' if is_valid else '需要重建'}")
    print(f"原因: {reason}")
    print(f"\n当前配置哈希: {get_current_config_hash()}")
    print(f"当前配置快照: {json.dumps(get_current_config_snapshot(), ensure_ascii=False, indent=2)}")

    if not is_valid:
        response = input("\n是否重建向量库? (y/n): ")
        if response.lower() == "y":
            delete_vector_store()
            print("\n已删除旧向量库。请运行: python rag/vector_store.py")
