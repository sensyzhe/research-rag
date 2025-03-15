from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings  # 新的导入方式

from langchain.tools.retriever import create_retriever_tool
import os

load_dotenv()



# ------------------------------------------------------------
from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import tools_condition
from langchain_core.messages import HumanMessage

def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    model = ChatOpenAI(
    temperature=0.7, # 降低随机性以获得更稳定的回答
    model="glm-4",
    openai_api_key="e046a661e0b44ed688c4d5c9c9940ff7.LXpzycHVawKsDebj",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    print(response)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}
# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
retrieve = ToolNode(tools=[retriever_tool])
workflow.add_node("retrieve", retrieve)  # retrieval

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,#如果上一个message带有tool call,导向toolnode，否则导向end
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)
workflow.add_edge("retrieve", "agent")

graph = workflow.compile()

graph.invoke({"messages": [HumanMessage(content="你好，请列出哮喘的症状")]})
