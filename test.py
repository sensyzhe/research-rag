from LLM import build_graph,run_graph
import os
import dotenv
# from ragas import EvaluationDataset
# from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from langchain_openai import ChatOpenAI



def get_response(question,graph):
    thread_id = "1"
    config = {"configurable": {"thread_id": thread_id}}
    print(question)
    
    context =   graph.invoke({"messages": [{"role": "user", "content": question}]},config).get("messages")
    response = context[-1].content
    tool_messages = [message for message in context if message.type == "tool"]
    tool_context = tool_messages[-1].content
    return response,tool_context

if __name__ == "__main__":
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    graph = build_graph()
    run_graph(graph,"你好")