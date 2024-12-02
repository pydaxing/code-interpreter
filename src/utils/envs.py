import os

## LangChain Config
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "lengmou-langgraph"

## Azure Config
os.environ["AZURE_OPENAI_API_KEY"] = "icbu-azure-algo"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://iai.alibaba-inc.com/azure"
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-03-01-preview"

## Tavily Config
os.environ["TAVILY_API_KEY"] = "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"

workdir = "/mnt/workspace/workgroup/lengmou/Demo/code-interpreter/src/"


sandbox_image_save_dir = os.path.join(workdir, "interact", "images")
upload_file_save_dir = os.path.join(workdir, "interact", "files")
sqlite_memory_save_dir = os.path.join(workdir, "interact", "sqlite")
prompt_file = os.path.join(workdir,"prompt.yaml")
