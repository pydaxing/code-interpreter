import os
import sys
prj_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(prj_root_path)
sys.path.append("..")

from typing import Annotated
from langchain_core.tools import tool
import requests
import json

from utils.util import image_bytes_to_png

@tool
def sandbox_execute(
    kernel_id: Annotated[str, "The kernel_id of the sandbox used to run Python code."],
    code: Annotated[str, "The python code to execute in the sandbox"]
):
    """Executing Python code, it is necessary to provide the unique identifier kernel_id of the sandbox and the python code to be executed"""
    try:
        url = "https://pybox.alibaba-inc.com/execute"
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "API-KEY": "pb-ImsGXLEM24bBa961701b4622A88fD2F56a53F4fAJzI633CX",
            "KERNEL-ID": kernel_id,
        }
        data = {
            "code": code
        }

        response = requests.post(url, headers=headers, data=json.dumps(data, ensure_ascii=False)).json()
        try:
            response = response["result"]
            response.pop("API_KEY")
            final_results = []
            if response["results"]:
                for result in response["results"]:
                    if result["type"] == "image/png":
                        image_path = image_bytes_to_png(result["data"])
                        final_results.append(image_path)
                    else:
                        final_results.append(result)
            response["results"] = final_results
            if response["error"]:
                response["error"].pop("traceback")
        except Exception as e:
            return f"Failed to execute code. Error: {repr(response)} "
    except Exception as e:
        return f"Failed to execute code. Error: {repr(e)}"
    return (
        response
    )