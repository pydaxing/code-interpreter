import os
from langchain_openai import AzureChatOpenAI

class Azure:
    def __new__(cls, emp_id, model_name, streaming=False, is_ptu=True):
        if is_ptu:
            headers = {"empId": emp_id, "useType":"ptuFirst"}
        else:
            headers = {"empId": emp_id}

        llm = AzureChatOpenAI(
            default_headers={"empId": emp_id, "useType":"ptuFirst"},
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_deployment=model_name,
            streaming=streaming
        )
        
        return llm