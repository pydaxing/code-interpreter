import os
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
from utils.envs import sqlite_memory_save_dir
from datetime import datetime


memory = AsyncSqliteSaver.from_conn_string(os.path.join(sqlite_memory_save_dir, f"checkpoints-{str(datetime.now())}.sqlite"))
