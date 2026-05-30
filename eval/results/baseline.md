# RAG 评估记录

用于对比不同配置下的检索效果，并作为简历数据来源。  
每次跑完 `python -m eval.run_rag_eval` 后，在下方新增一节或更新对应实验。

---

## baseline（当前默认配置）

| 项 | 值 |
|----|-----|
| 日期 | 2026-05-29 |
| 命令 | `D:\miniconda\envs\RAG_env\python.exe -m eval.run_rag_eval` |
| 评测集 | `eval/datasets/rag_eval.json`（15 条） |
| 报告文件 | [rag_eval_20260529_163044.json](./rag_eval_20260529_163044.json) |

### 配置快照

| 参数 | 值 |
|------|-----|
| embedding_model | text-embedding-v4 |
| chunk_size | 200 |
| chunk_overlap | 20 |
| k（检索条数） | 3 |
| similarity_threshold | 0.5 |

### 检索指标（写简历用）

| 指标 | 结果 |
|------|------|
| **Hit@3** | **60.0%**（9 / 15） |
| **MRR@3** | **0.4667** |
| 平均检索耗时 | 866.62 ms |

### Ragas 生成指标

未跑。需要时执行：

```powershell
python -m eval.run_rag_eval --with-ragas
```

跑完后把 `faithfulness`、`answer_relevancy`、`context_precision` 填到本节。

### 未命中样本（便于后续优化）

| ID | 问题 |
|----|------|
| q02 | 充电触点应该怎么清洁？ |
| q05 | 大户型选购扫地机器人续航应该选多久？ |
| q08 | 拖地后地面有明显水痕怎么解决？ |
| q09 | 边刷磨损到什么程度需要更换？ |
| q11 | 清扫时机器人频繁暂停是什么原因？ |
| q12 | 激光导航和视觉导航有什么区别？ |

可尝试：调低 `--threshold 0.25`、增大 `k`、调 `chunk_size`，或补充 `reference_contexts` 与知识库原文对齐。

### 简历表述（可直接改数字）

```
构建 15 条扫地机器人领域 RAG 评测集，实现 Hit@3 / MRR 自动化评估；
在 chunk_size=200、k=3 配置下，检索 Hit@3 为 60.0%，MRR@3 为 0.47。
```

---

## exp_rewrite_rerank（query 改写 + embedding 重排）

> 已在 `config/rag.yml` 开启 `enable_query_rewrite` / `enable_rerank` 后重跑：

```powershell
python -m eval.run_rag_eval
```

把 Hit@3、MRR 填到下面，与 baseline 对比。

| 指标 | baseline | exp_rewrite_rerank |
|------|----------|---------------------|
| Hit@3 | 60.0% | |
| MRR@3 | 0.4667 | |

---

## exp1（待填写：例如 chunk_size=400）

> 改 `config/chroma.yml` 后重新跑评估，把下面表格填完整。

| 项 | 值 |
|----|-----|
| 日期 | |
| 命令 | `python -m eval.run_rag_eval` |
| 报告文件 | `eval/results/rag_eval_YYYYMMDD_HHMMSS.json` |

### 配置变更

| 参数 | baseline | exp1 |
|------|----------|------|
| chunk_size | 200 | |
| chunk_overlap | 20 | |
| k | 3 | |

### 检索指标

| 指标 | baseline | exp1 | 变化 |
|------|----------|------|------|
| Hit@3 | 60.0% | | |
| MRR@3 | 0.4667 | | |
| 平均检索耗时 (ms) | 866.62 | | |

### 简历对比句（有 exp1 数据后再写）

```
通过评测集对比分片策略，chunk_size 由 200 调整为 ___ 后，Hit@3 由 60.0% 提升至 ___%。
```

---

## exp2（待填写：可选实验）

例如：`k=5`、`--threshold 0.25`、混合检索等。格式同 exp1。

---

## 操作备忘

```powershell
conda activate RAG_env
cd D:\develop\my_project\RAG_Project2

# 仅检索评估（快）
D:\miniconda\envs\RAG_env\python.exe -m eval.run_rag_eval

# 含 Ragas（慢，耗 API）
D:\miniconda\envs\RAG_env\python.exe -m eval.run_rag_eval --with-ragas

# 调相似度阈值
D:\miniconda\envs\RAG_env\python.exe -m eval.run_rag_eval --threshold 0.25
```

每次跑完：复制终端摘要 → 更新本节 → 保留对应 `rag_eval_*.json` 勿删。
