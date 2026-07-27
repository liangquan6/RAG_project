import os
import hashlib
import re
from utils.logger_handler import logger
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


def get_file_md5_hash(file_path: str):   # 获取文件的md5值
    if not os.path.exists(file_path):
        logger.error(f"[md5计算] 文件{file_path}不存在")
        return

    if not os.path.isfile(file_path):
        logger.error(f"[md5计算] 文件{file_path}不是文件")
        return

    md5_obj = hashlib.md5()

    chunk_size = 4096      # 4KB分片，避免文件过大导致内存占用过高

    try:
        with open(file_path, "rb") as f:   # 必须以二进制模式打开，避免文件编码问题
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)

            """
            chunk = f.read(chunk_size)
            while chunk:
                md5_obj.update(chunk)
                chunk = f.read(chunk_size)
            """
            md5_hash = md5_obj.hexdigest()
            return md5_hash
    except Exception as e:
        logger.error(f"[md5计算] 文件{file_path}计算失败: {str(e)}")
        return None


def listdir_with_allowed_type(path: str, allowed_types: tuple[str]):     # 返回指定类型文件的列表
    files = []

    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type] 路径{path}不是文件夹")
        return ()

    for file in os.listdir(path):
        if file.endswith(allowed_types):
            files.append(os.path.join(path, file))

    return tuple(files)

def pdf_loader(file_path: str,password: str = None) -> list[Document]:    # 加载pdf文件
    return PyPDFLoader(file_path, password=password).load()


def txt_loader(file_path: str) -> list[Document]:    # 加载txt文件
    return TextLoader(file_path, encoding='utf-8').load()