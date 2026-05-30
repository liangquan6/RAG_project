from utils.config_handler import prompts_config
from utils.path_tools import get_abs_path
from utils.logger_handler import logger


def load_system_prompts():
    try:
        system_prompt_path = get_abs_path(prompts_config["main_prompt_path"])
    except Exception as e:
        logger.error("[load_system_prompt]在yaml配置项中没有找到main_prompt_path")
        raise e

    try:
        return open(system_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_system_prompt]读取系统提示词文件失败: {str(e)}")
        raise e


def load_rag_prompts():
    try:
        rag_prompt_path = get_abs_path(prompts_config["rag_summary_prompt_path"])
    except Exception as e:
        logger.error("[load_rag_prompt]在yaml配置项中没有找到rag_summary_prompt_path")
        raise e

    try:
        return open(rag_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_rag_prompt]读取RAG总结提示词文件失败: {str(e)}")
        raise e


def load_report_prompts():
    try:
        report_prompt_path = get_abs_path(prompts_config["report_prompt_path"])
    except Exception as e:
        logger.error("[load_report_prompt]在yaml配置项中没有找到report_prompt_path")
        raise e

    try:
        return open(report_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_report_prompt]读取报告生成提示词文件失败: {str(e)}")
        raise e


if __name__ == "__main__":
    print(load_system_prompts())
    print(load_rag_prompts())
    print(load_report_prompts())
