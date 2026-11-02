# SPDX-License-Identifier: Apache-2.0
from langchain_core.tools import BaseTool

from app.tools.calculator import calculate
from app.tools.datetime_tool import current_datetime
from app.tools.fetch_page import fetch_page
from app.tools.memory_tools import recall, remember
from app.tools.web_search import web_search

ALL_TOOLS: list[BaseTool] = [
    web_search,
    fetch_page,
    remember,
    recall,
    current_datetime,
    calculate,
]

CHAT_MODE_TOOLS: list[BaseTool] = [
    remember,
    recall,
    current_datetime,
    calculate,
]
