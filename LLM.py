from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage


from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition
from dotenv import load_dotenv
from tools import get_retriever_tool

# 加载环境变量
# load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

# 获取检索工具
tool = get_retriever_tool()
tools = [tool]

# 定义聊天机器人函数，处理状态并返回消息
def chatbot(state: State):
    llm = ChatOpenAI(
        model="glm-4",
        openai_api_key="e046a661e0b44ed688c4d5c9c9940ff7.LXpzycHVawKsDebj",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
    )
    llm_with_tools = llm.bind_tools(tools)
    return  {"messages": [llm_with_tools.invoke(state["messages"])]}

# 运行图形流程，处理用户输入并输出结果
def run_graph(graph,input,thread_id=1):
    thread_id = str(thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    print("\nAssistant: ")
    for message_chunk, metadata in graph.stream(
        {"messages": [{"role": "user", "content": input}]},
        config, 
        stream_mode= "messages"
        ):
        if message_chunk.content:
            if message_chunk.type == 'tool':
                print("\ntool:")
                print(message_chunk.content, end="\nAssistant: ", flush=True)
            else:
                print(message_chunk.content, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    graph_builder = StateGraph(State)

    graph_builder.add_node("chatbot", chatbot)

    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)

    # 添加条件边，根据消息是否包含工具调用决定流向
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,#如果上一个message带有tool call,导向toolnode，否则导向end
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.set_entry_point("chatbot")

    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    # 进入主循环，处理用户输入
    while True:
        try:
            user_input = input("User: \n")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            run_graph(graph,user_input)
        except:
            # 如果 input() 不可用，使用默认问题
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            run_graph(graph,user_input)
            break
