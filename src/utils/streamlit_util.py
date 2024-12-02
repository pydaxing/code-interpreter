import re
import uuid
import json
import time
import streamlit as st
import itertools
import functools
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import END, StateGraph
from PIL import Image


from tools.create_sandbox import create_sandbox
from tools.close_sandbox import close_sandbox
from tools.sandbox_execute import sandbox_execute
from components.agent import AIHelper
from components.state import AgentState
from components.memory import memory
from components.router import aihelper_router
from llms.azure import Azure


def clear_chat_history(unique_session_id):
    st.session_state.threads[unique_session_id] = str(uuid.uuid4())
    st.session_state.files[unique_session_id] = []



async def create_agent_node(state, config: RunnableConfig, agent, name):
    result = await agent.ainvoke(state["messages"], config)
    return {
        "messages": [result]
    }


@st.cache_resource
def build_graph(model_name):
    tools = [create_sandbox, sandbox_execute, close_sandbox]
    llm = Azure("337310", model_name, streaming=True, is_ptu=True)
    aihelper_agent = AIHelper(llm, tools).get_agent()

    aihelper_agent_node = functools.partial(create_agent_node, agent=aihelper_agent, name="AIHelper")
    tool_node = ToolNode(tools)

    workflow = StateGraph(AgentState)
    workflow.add_node("AIHelper", aihelper_agent_node)
    workflow.add_node("tools", tool_node)

    workflow.add_conditional_edges("AIHelper", tools_condition)
    workflow.add_edge("tools", "AIHelper")
    workflow.set_entry_point("AIHelper")

    graph = workflow.compile(checkpointer=memory)
    return graph



async def history(graph, config):
    state_snapshot = await graph.aget_state(config)
    history = state_snapshot.values["messages"]
    return history


def on_tool_start(event):
    if event['name'] == "create_sandbox":
        result = "\n\n开始创建Python环境\n\n"
    elif event['name'] == "close_sandbox":
        result = "\n\n开始释放Python环境\n\n"
    elif event['name'] == "sandbox_execute":
        code = event['data'].get('input')["code"]
        result = "\n\n开始执行Python代码\n\n" + f"```python\n{code}\n```\n\n"
    for token in result:
        yield token
        time.sleep(0.001)
        
        
def on_tool_end(event):
    if event['name'] == "create_sandbox":
        result = f"\n\nPython环境已创建: {event['data'].get('output')}\n\n"
    elif event['name'] == "close_sandbox":
        result = f"\n\nPython环境已释放: {event['data'].get('output')}\n\n"
    elif event['name'] == "sandbox_execute":
        exec_out = event['data'].get('output')
        if not isinstance(exec_out, str):
            print(exec_out)

            result = ""
            if exec_out["results"]:
                for res in exec_out["results"]:
                    st.image(image=res, caption="")

            if exec_out["logs"]["stdout"]:
                newline = "\n"
                result = f"\n\nPython执行完成: \n\n```\n{newline.join(exec_out['logs']['stdout'])}\n```\n\n"
                if not exec_out["results"]:
                    pattern = r"\/mnt\/workspace\/.*(\.png|\.jpg|\.jpeg)"
                    for line in exec_out["logs"]["stdout"]:
                        match = re.search(pattern, line)
                        if match:
                            st.image(image=match.group(), caption="")

            if exec_out["error"]:
                result = f"\n\nPython执行完成: \n\n```\n{exec_out['error']['name'] + ': ' + exec_out['error']['value']}\n```\n\n"
        else:
            result = f"\n\nPython执行完成: \n\n```\n{exec_out}\n\n"
    for token in result:
        yield token
        time.sleep(0.001)


async def solve(graph, config, prompt):
    response = ""
    pre_kind = "on_chat_model_stream"
    with st.chat_message("assistant"):
        with st.spinner():
            placeholder = st.empty()
            async for event in graph.astream_events({"messages": [HumanMessage(content=prompt)]}, config, version="v1"):
                kind = event["event"]
                if kind == "on_chat_model_stream":
                    if kind != pre_kind:
                        placeholder = st.empty()
                        response = ""
                    content = event["data"]["chunk"].content
                    if content:
                        response += content
                        placeholder.markdown(response)
                    pre_kind = "on_chat_model_stream"
                elif kind == "on_tool_start":
                    if kind != pre_kind:
                        placeholder = st.empty()
                        response = ""
                    for token in on_tool_start(event):
                        response += token
                        placeholder.markdown(response)
                    pre_kind = "on_tool_start"
                elif kind == "on_tool_end":  
                    if kind != pre_kind:
                        placeholder = st.empty()
                        response = ""
                    for token in on_tool_end(event):
                        response += token
                        placeholder.markdown(response)
                    pre_kind = "on_tool_end"

                    
def tool_message_display(message):
    if message.dict()["name"] == "create_sandbox":
        result = f"\n\nPython环境已创建\n\n{message.content}"
    elif message.dict()["name"] == "close_sandbox":
        result = f"\n\nPython环境已释放\n\n{message.content}"
    elif message.dict()["name"] == "sandbox_execute":
        exec_out = json.loads(message.content)
        result = "\n\nPython执行完成: \n\n"
        if exec_out["results"]:
            for res in exec_out["results"]:
                st.image(image=res, caption="")
                result += f"\n{res}\n"
        if exec_out["logs"]["stdout"]:
            newline = "\n"
            result += f"```\n{newline.join(exec_out['logs']['stdout'])}\n```\n\n"
            if not exec_out["results"]:
                pattern = r"\/mnt\/workspace\/.*(\.png|\.jpg|\.jpeg)"
                for line in exec_out["logs"]["stdout"]:
                    match = re.search(pattern, line)
                    if match:
                        st.image(image=match.group(), caption="")
        if exec_out["error"]:
            result += f"\n\n```\n{exec_out['error']['name'] + ': ' + exec_out['error']['value']}\n```\n\n"
    return result


def ai_message_display(message):
    result = ""
    if message.tool_calls:
        if message.content != "":
            result += message.content
        if message.tool_calls[0]["name"] == "create_sandbox":
            result += "开始创建Python环境\n"
        elif message.tool_calls[0]["name"] == "close_sandbox":
            result += "开始释放Python环境\n"
        elif message.tool_calls[0]["name"] == "sandbox_execute":
            code = message.tool_calls[0]["args"]["code"]
            result += f"开始执行Python代码\n\n```python\n{code}\n```\n\n"
    else:                
        result = message.content
    return result