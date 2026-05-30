"""
智扫通机器人智能客服 - Streamlit 前端应用

功能概述：
- 提供友好的聊天界面
- 维护对话历史记录
- 调用 ReactAgent 进行智能交互
- 支持流式输出，提升用户体验
"""

import streamlit as st
from agent.react_agent import ReactAgent


def _init_messages() -> None:
    """初始化对话历史（兼容旧版 session_state 键名 message）。"""
    if "messages" in st.session_state:
        return
    if "message" in st.session_state:
        st.session_state.messages = st.session_state.pop("message")
    else:
        st.session_state.messages = []


def merge_stream_chunks(chunks: list[str]) -> str:
    """
    合并流式输出片段。

    Agent 的 stream_mode=values 可能每次产出整段快照，也可能产出增量块；
    若最后一段已覆盖绝大部分内容，则取最后一段，否则拼接全部片段。
    """
    if not chunks:
        return ""
    joined = "".join(chunks).strip()
    last = chunks[-1].strip()
    if last and len(last) >= len(joined) * 0.8:
        return last
    return joined


def _init_agent() -> None:
    if "agent" in st.session_state:
        return
    with st.spinner("正在加载 Agent（含知识库检查，首次可能较慢）..."):
        st.session_state.agent = ReactAgent()


# ==========================================
# 页面配置
# ==========================================
st.set_page_config(
    page_title="智扫通智能客服",
    page_icon="🤖",
    layout="centered",
)

st.title("智扫通机器人智能客服")
st.caption("RAG 知识问答 · 多轮对话 · ReAct Agent · 使用报告生成")

with st.sidebar:
    st.markdown("**操作**")
    if st.button("清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

_init_messages()

try:
    _init_agent()
except Exception as e:
    st.error(f"Agent 初始化失败：{e}")
    st.info("请确认已设置 `DASHSCOPE_API_KEY`，并使用 RAG_env 环境启动。")
    st.stop()

# ==========================================
# 渲染历史对话
# ==========================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 用户输入处理
# ==========================================
prompt = st.chat_input("请输入您的问题…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    chunks: list[str] = []

    def stream_with_cache():
        # 传入完整会话（含本轮 user），供 Agent 多轮记忆与 RAG 改写
        chat_history = list(st.session_state.messages)
        for chunk in st.session_state.agent.execute_stream(prompt, history=chat_history):
            if chunk:
                chunks.append(chunk)
                yield chunk

    with st.chat_message("assistant"):
        try:
            with st.spinner("智能客服思考中..."):
                st.write_stream(stream_with_cache)
            reply = merge_stream_chunks(chunks)
            if not reply:
                reply = "抱歉，未能生成有效回复，请换个方式提问或稍后重试。"
        except Exception as e:
            reply = f"抱歉，服务出现异常：{e}"
            st.error(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
