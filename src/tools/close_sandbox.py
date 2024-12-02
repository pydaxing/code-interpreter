from typing import Annotated
from langchain_core.tools import tool
import requests

@tool
def close_sandbox(
    kernel_id: Annotated[str, "The kernel id of the sandbox that need to be closed"]
):
    """Shutting down the Python Sandbox, it is necessary to close the sandbox and release resources after the user's task is completed. This requires passing the unique identifier, kernel_id, of the Sandbox"""
    try:
        url = "https://pybox.alibaba-inc.com/close"
        headers = {
            "API-KEY": "pb-ImsGXLEM24bBa961701b4622A88fD2F56a53F4fAJzI633CX",
            "KERNEL-ID": kernel_id
        }

        response = requests.request("GET", url, headers=headers).json()
    except Exception as e:
        return f"Failed to Close a Python SandBox. Error: {repr(e)}"
    return (
        f"Python Sandbox {kernel_id} closed successfully."
    )