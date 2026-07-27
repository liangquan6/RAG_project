# 智扫通机器人 Agent 客服

基于 **RAG + ReAct 多工具编排** 的扫地机器人智能客服 Demo。支持知识库问答、天气/定位查询、用户使用报告生成，以及多轮对话与 RAG 效果评估。

## 功能概览

| 模块 | 说明 |
|------|------|
| **ReAct Agent** | 7 个 Tool + Middleware（监控、日志、报告场景动态 Prompt） |
| **RAG** | Chroma 向量库、通义 Embedding、Query 改写、Embedding 重排序 |
| **多轮对话** | 滑动窗口记忆（默认最近 10 条消息） |
| **流式输出** | 真逐 token 流式（`stream_mode="messages"` + 防 ToolMessage 泄露） |
| **向量库治理** | 配置变更自动检测并重建，MD5 增量入库，文件更新自动清理旧分片 |
| **评估** | Hit@5 / MRR 检索 + Ragas 生成（Faithfulness / Answer Relevancy / Context Precision） |

## 项目结构

```
RAG_Project2/
├── app.py                 # Streamlit 前端
├── agent/                 # ReAct Agent 与 Tools
├── rag/                   # RAG 检索、改写、重排
├── model/                 # 大模型 / Embedding 工厂
├── utils/                 # 配置、日志、记忆、API
├── config/                # YAML 配置
├── prompts/               # 系统 / RAG / 报告提示词
├── data/                  # 知识库文档（pdf/txt）
├── eval/                  # RAG 评估脚本与评测集
└── requirements.txt
```

## 环境要求

- Python **3.10+**
- 推荐使用 Conda 环境（如 `RAG_env`）
- [通义千问 DashScope API Key](https://help.aliyun.com/zh/model-studio/)

## 快速开始

### 1. 克隆并进入项目

```bash
cd RAG_Project2
```

### 2. 创建环境并安装依赖

```bash
conda create -n RAG_env python=3.10 -y
conda activate RAG_env
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

编辑 `.env`，填入：

```env
DASHSCOPE_API_KEY=你的密钥
```

也可在系统环境变量中设置 `DASHSCOPE_API_KEY`。

### 4. 启动服务

**务必在项目根目录执行，并使用当前环境的 Python：**

```bash
# 推荐
python -m streamlit run app.py

# Windows 示例（conda 环境绝对路径）
# D:\miniconda\envs\RAG_env\python.exe -m streamlit run app.py
```

浏览器访问：**http://localhost:8501**

> 首次启动会进行向量库检查与知识库入库，可能需 1~5 分钟。

### 5. 运行 RAG 评估

```bash
# 仅检索指标（Hit@5、MRR，较快）
python -m eval.run_rag_eval

# 含 Ragas 生成指标（较慢，消耗更多 API）
python -m eval.run_rag_eval --with-ragas
```

最新评测结果（15 样本，qwen-plus）：

| 指标 | 数值 |
|------|------|
| Hit@5 | 66.67% |
| MRR@5 | 0.5333 |
| Faithfulness | 0.9222 |
| Answer Relevancy | 0.7780 |
| Context Precision | 0.6967 |

结果保存在 `eval/results/rag_eval_*.json`。

## 配置说明

### `config/rag.yml`

| 配置项 | 说明 |
|--------|------|
| `chat_model_name` | 对话模型，当前 `qwen-plus`（流式 + 工具调用稳定） |
| `embedding_model_name` | 向量模型，如 `text-embedding-v4` |
| `enable_query_rewrite` | 是否开启 Query 多表述改写 |
| `enable_rerank` | 是否开启 Embedding 重排序 |
| `retrieve_k` | 粗召回条数（重排前） |

### `config/chroma.yml`

| 配置项 | 说明 |
|--------|------|
| `k` | 最终送入 LLM 的检索条数 |
| `chunk_size` / `chunk_overlap` | 文档分片参数 |
| `data_path` | 知识库目录 |

### `config/agent.yml`

| 配置项 | 说明 |
|--------|------|
| `conversation.max_messages` | 多轮对话滑动窗口大小 |
| `weather_api` | 天气服务（mock / 心知 / 和风） |
| `location_api` | 定位服务（mock / ip_api） |

## 架构示意

```
用户 (Streamlit)
    │
    ▼
ReactAgent (ReAct + Middleware)
    ├── rag_summarize ──► RAG（改写 → 检索 → Rerank → 总结）
    ├── get_weather / get_user_location
    ├── get_user_id / get_current_month / fetch_external_data
    └── fill_context_for_report（触发报告 Prompt）
    │
    ▼
Chroma 向量库 ← data/*.txt,pdf
```

## Mock 与真实能力说明

| 能力 | 默认行为 |
|------|----------|
| 大模型 / Embedding | 通义 DashScope（需 API Key） |
| 天气 | 可在 `agent.yml` 配置 mock 或真实 API |
| 定位 | 默认 IP 定位；与**会话记忆**无关 |
| 用户 ID / 月份 | 演示用随机 mock |
| 使用报告数据 | `data/external/records.csv` |

**多轮记忆**：仅在当前会话未清空时有效；点击「清空对话」后历史不再保留。

## 常见问题

### `No module named 'xxx'`

确保已 `conda activate RAG_env`，并在**项目根目录**运行命令。

### 换 Embedding 后服务异常

删除 `chroma_db/` 与 `md5.text` 后重启，或依赖自动版本检测重建（见 `utils/vector_store_manager.py`）。

### Streamlit 断连

不要用 `python app.py`，请使用 `python -m streamlit run app.py`。

### 评估 Hit@3 偏低

可调整 `chunk_size`、`k`，或运行 `python -m eval.run_rag_eval --threshold 0.25`，详见 `eval/results/baseline.md`。

## 技术栈

Python · LangChain · LangGraph · Chroma · 通义千问 · DashScope Embedding · Streamlit · Ragas

## License

仅供学习与面试作品展示使用。
