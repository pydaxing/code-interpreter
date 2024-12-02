from typing import Literal

def aihelper_router(state) -> Literal["tools", "__end__"]:
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        # The previous agent is invoking a tool
        return "tools"
    if "TO USER" in last_message.content:
        # Any agent decided the work is done
        return "__end__"