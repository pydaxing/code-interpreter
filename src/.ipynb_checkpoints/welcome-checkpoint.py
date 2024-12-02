import os
import sys
import re
prj_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(prj_root_path)
sys.path.append("..")

import streamlit as st
import asyncio
import uuid
import json
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

from utils.envs import upload_file_save_dir
from utils.streamlit_util import clear_chat_history, build_graph, history, solve
from utils.streamlit_util import tool_message_display, ai_message_display



st.set_page_config(layout="wide", page_title="Python Interpreter", page_icon='🦙')
st.title('Python Interpreter')

# 模型选择器
selected_llm = st.selectbox(label="Select Interpreter Model", options=["gpt-4-turbo", "gpt-4o"], key=f"interpreter_model", label_visibility="collapsed")
# 唯一会话id
if "unique_session_id" not in st.session_state.keys():
    st.session_state["unique_session_id"] = str(uuid.uuid4())
if "threads" not in st.session_state.keys():
    st.session_state["threads"] = {}
    st.session_state.threads[st.session_state["unique_session_id"]] = str(uuid.uuid4())
if "files" not in st.session_state.keys():
    st.session_state["files"] = {}
    st.session_state.files[st.session_state["unique_session_id"]] = []

# 设置sidebar
with st.sidebar:
    st.markdown("## Upload a File")
    uploaded_files = st.file_uploader("Choose a file", label_visibility="collapsed", accept_multiple_files=True)
    
    st.session_state.files[st.session_state["unique_session_id"]] = []
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            file_content = uploaded_file.read()
            save_path = os.path.join(upload_file_save_dir, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(file_content)
            st.session_state.files[st.session_state["unique_session_id"]].append(save_path)
    
    st.button('Clear Chat History', use_container_width=True, on_click=clear_chat_history, kwargs={"unique_session_id": st.session_state["unique_session_id"]})
    
    
graph = build_graph(selected_llm)
config = {"configurable": {"thread_id": st.session_state.threads[st.session_state["unique_session_id"]]}, "recursion_limit": 100}

histories = asyncio.run(history(graph, config))

user_input = False
for message in histories:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content.split(", 文件路径列表：")[0])
        user_input = True
    if user_input:
        assistant = st.chat_message("assistant")
        user_input = False
    if isinstance(message, ToolMessage):
        with assistant:
            result = tool_message_display(message)
            st.write(result)
    if isinstance(message, AIMessage):
        with assistant:
            result = ai_message_display(message)
            st.write(result)

if query := st.chat_input("Enter"):
    if st.session_state.files[st.session_state['unique_session_id']]:
        prompt = f"{query}, 文件路径列表：{str(st.session_state.files[st.session_state['unique_session_id']])}"
    else:
        prompt = query
    with st.chat_message("user"):
        st.markdown(query)
    asyncio.run(solve(graph, config, prompt))
