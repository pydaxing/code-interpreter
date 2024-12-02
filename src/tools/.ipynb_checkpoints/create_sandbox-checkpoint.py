from langchain_core.tools import tool
import requests

@tool
def create_sandbox():
    """To create a sandbox for executing Python, this tool must be called first to create a Python Sandbox before calling the tool that executes the code."""
    try:
        url = "https://pybox.alibaba-inc.com/create"
        headers = {
            "API-KEY": "pb-ImsGXLEM24bBa961701b4622A88fD2F56a53F4fAJzI633CX",
        }

        response = requests.request("GET", url, headers=headers).json()
    except Exception as e:
        return f"Failed to create a Python SandBox. Error: {repr(e)}"
    return (
        f"Python Sandbox created successfully. The kernel ID is {response['kernel_id']}."
    )