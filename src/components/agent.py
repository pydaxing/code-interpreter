import os
import sys
prj_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(prj_root_path)
sys.path.append("..")

import yaml
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from utils.envs import prompt_file
from .state import AgentState


class AIHelper:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.tool_names = ", ".join([tool.name for tool in tools])
        self.current_date = datetime.now()
        
        self.compose_prompt()
        self.build_agent()
    
    def compose_prompt(self):
        with open(prompt_file, 'r') as file:
            system_prompt = yaml.safe_load(file)["system_prompt"]
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
    
    def build_agent(self):
        self.prompt = self.prompt.partial(current_date=self.current_date)
        self.prompt = self.prompt.partial(tool_names=self.tool_names)
        self.agent = self.prompt | self.llm.bind_tools(self.tools)
        
    
    def get_agent(self):
        return self.agent
        