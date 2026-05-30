
"""
向量存储服务模块

功能概述：
- 管理 Chroma 向量数据库
- 文档加载与分片
- 基于 MD5 的增量加载与去重
- 提供检索器接口
"""

from langchain_chroma import Chroma
from utils.config_handler import chroma_config
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tools import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hash
from utils.logger_handler import logger
from utils.vector_store_manager import check_and_rebuild_if_needed, save_config_metadata
from langchain_core.documents import Document
import os


class VectorStoreService:
    """
    向量存储服务类

    主要功能：
    1. 初始化 Chroma 向量数据库
    2. 文档加载、分片、向量化存储
    3. MD5 去重，支持增量加载
    4. 提供检索器接口
    """

    def __init__(self, auto_load: bool = True):
        """
        初始化向量存储服务

        配置：
        - Chroma 集合名称
        - 嵌入模型
        - 持久化存储目录
        - 文本分片器

        参数:
            auto_load: 是否在初始化时检查版本并加载/增量更新知识库
        """
        needs_rebuild = check_and_rebuild_if_needed()
        if needs_rebuild:
            logger.info("[向量库]旧数据已清理，将重新入库")

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_config["chunk_size"],
            chunk_overlap=chroma_config["chunk_overlap"],
            separators=chroma_config["separators"],
            length_function=len,
        )

        self.vector_store = Chroma(
            collection_name=chroma_config["collection_name"],
            embedding_function=embed_model,
            persist_directory=chroma_config["persist_directory"],
        )

        if auto_load:
            self.load_document()

    def get_retriever(self, k: int | None = None):
        """
        获取向量检索器

        参数:
            k: 检索条数，默认使用 chroma.yml 中的 k

        返回:
            LangChain Retriever 对象，用于根据查询检索相关文档
        """
        top_k = k if k is not None else chroma_config["k"]
        return self.vector_store.as_retriever(search_kwargs={"k": top_k})

    def load_document(self):
        """
        从数据文件夹加载文档并向量化存储

        流程：
        1. 遍历数据文件夹
        2. 计算文件 MD5，检查是否已加载
        3. 加载文件内容（PDF/TXT）
        4. 文本分片
        5. 向量化存储
        6. 记录 MD5，避免重复加载

        返回:
            None
        """

        def check_md5_hash(md5_for_check: str):
            """
            检查 MD5 是否已存在（用于去重）

            参数:
                md5_for_check: 待检查的 MD5 哈希值

            返回:
                True - 已存在；False - 不存在
            """
            md5_path = get_abs_path(chroma_config["md5_hex_store"])

            if not os.path.exists(md5_path):
                with open(md5_path, "w", encoding="utf-8") as f:
                    pass
                return False

            with open(md5_path, "r", encoding="utf-8") as f:
                existing_hashes = {line.strip() for line in f if line.strip()}
                return md5_for_check in existing_hashes

        def save_md5_hash(md5_for_save: str):
            """
            保存 MD5 哈希值到记录文件

            参数:
                md5_for_save: 要保存的 MD5 哈希值
            """
            with open(get_abs_path(chroma_config["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_save + "\n")

        def get_file_documents(read_path: str):
            """
            根据文件扩展名加载文档

            参数:
                read_path: 文件路径

            返回:
                Document 列表
            """
            if read_path.endswith(".pdf"):
                return pdf_loader(read_path)

            if read_path.endswith(".txt"):
                return txt_loader(read_path)

            return []

        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_config["data_path"]),
            tuple(chroma_config["allow_knowledge_file_type"])
        )

        if not allowed_files_path:
            logger.warning("[加载知识库]数据文件夹为空或未找到允许的文件类型")
            return

        loaded_count = 0
        skipped_count = 0
        error_count = 0

        for path in allowed_files_path:
            md5_hex = get_file_md5_hash(path)

            if check_md5_hash(md5_hex):
                logger.info(f"[加载知识库]{path}内容已存在于知识库，跳过")
                skipped_count += 1
                continue

            try:
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内容为空，跳过")
                    skipped_count += 1
                    continue

                split_document: list[Document] = self.splitter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后内容为空，跳过")
                    skipped_count += 1
                    continue

                self.vector_store.add_documents(split_document)
                save_md5_hash(md5_hex)

                loaded_count += 1
                logger.info(f"[加载知识库]{path}内容已加载 ({len(split_document)} 个片段)")

            except Exception as e:
                error_count += 1
                logger.error(f"[加载知识库]{path}内容加载失败: {str(e)}", exc_info=True)
                continue

        save_config_metadata()
        logger.info(
            f"[加载知识库]完成: 加载 {loaded_count} 个文件, "
            f"跳过 {skipped_count} 个, 失败 {error_count} 个"
        )


if __name__ == "__main__":
    vector_store_service = VectorStoreService()
    retriever = vector_store_service.get_retriever()
    results = retriever.invoke("你好")
    for r in results:
        print(r.page_content)
        print("-" * 20)
