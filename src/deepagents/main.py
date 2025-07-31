from deepagents.graph import create_deep_agent
from dotenv import load_dotenv
from typing import Literal
from tavily import TavilyClient
import os

load_dotenv("../.env")

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    tavily_async_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return tavily_async_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


research_instructions = """You are an expert researcher. Your job is to conduct thorough research, and then write a polished report.

You have access to a few tools.

## `internet_search`

Use this to run an internet search for a given query. You can specify the number of results, the topic, and whether raw content should be included.
Only call this tool up to 5 times. Don't rabbit hole.

While doing this, you have access to several tools that allow you access and write information to a remote filesystem. This can be particularly useful for storing information that you find during your research.
The filesystem is a great tool for storing a lot of information without bloating your own context window too many tokens. This way, you can later search for and read files when you need to, but you don't have to have to keep all information in your context window.
Here are some usage patterns:
- Before conducting deep research, you should check the filesystem to see if there is already information that you can use.
- You can use the filesystem to store raw information that you find during research, for future reference and consumption.
- You can use the filesystem to store information that you have written, for future reference and consumption.
- After conducting research, you should write the report to the filesystem to save your work.
"""

deep_agent = create_deep_agent(
    instructions=research_instructions,
    tools=[internet_search],
)