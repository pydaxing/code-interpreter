import os
import sys
prj_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(prj_root_path)
sys.path.append("..")

import functools
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph

from tools.create_sandbox import create_sandbox
from tools.close_sandbox import close_sandbox
from tools.sandbox_execute import sandbox_execute
from components.agent import AIHelper
from components.state import AgentState
from components.memory import memory
from components.router import aihelper_router
from llms.azure import Azure



tools = [create_sandbox, sandbox_execute, close_sandbox]

llm = Azure("337310", "gpt-4-turbo", is_ptu=True)

aihelper_agent = AIHelper(llm, tools)

def create_agent_node(state, agent, name):
    result = agent(state)
    # We convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result]
    }


aihelper_agent_node = functools.partial(create_agent_node, agent=aihelper_agent, name="AIHelper")
tool_node = ToolNode(tools)

workflow = StateGraph(AgentState)
workflow.add_node("AIHelper", aihelper_agent_node)
workflow.add_node("tools", tool_node)

workflow.add_conditional_edges("AIHelper", aihelper_router, {"tools": "tools", "__end__": END})
workflow.add_edge("tools", "AIHelper")

workflow.set_entry_point("AIHelper")

graph = workflow.compile(checkpointer=memory)



config = {"configurable": {"thread_id": "1"}, "recursion_limit": 100}
while True:
    user_input = input("User: ")
    for event in graph.stream({"messages": [("user", user_input)]}, config):
        for value in event.values():
            if isinstance(value["messages"][-1], BaseMessage):
                print(value["messages"][-1].name,": ",  value["messages"][-1].content)