from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition
from dotenv import load_dotenv
from tools import get_retriever_tool

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]



graph_builder = StateGraph(State)


# tool = TavilySearchResults(max_results=2)
tools = get_retriever_tool()
llm = ChatOpenAI(
    temperature=0.7, # 降低随机性以获得更稳定的回答
    model="glm-4",
    openai_api_key="e046a661e0b44ed688c4d5c9c9940ff7.LXpzycHVawKsDebj",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return  {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,#如果上一个message带有tool call,导向toolnode，否则导向end
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)



# The config is the **second positional argument** to stream() or invoke()!
def run_graph(thread_id,input):
    thread_id = str(thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    print("Assistant: ")
    for message_chunk, metadata in graph.stream(
        {"messages": [{"role": "user", "content": input}]},
        config, 
        stream_mode= "messages"
        ):
        if message_chunk.content:
            print(message_chunk.content, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    while True:
        try:
            user_input = input("User: \n")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            run_graph(1,user_input)
        except:
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            run_graph(1,user_input)
            break
