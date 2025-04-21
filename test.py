from LLM import build_graph,run_graph
import os
import dotenv
# from ragas import EvaluationDataset
# from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from langchain_openai import ChatOpenAI
import math


def get_response(question,graph):
    thread_id = "1"
    config = {"configurable": {"thread_id": thread_id}}
    print(question)
    
    context =   graph.invoke({"messages": [{"role": "user", "content": question}]},config).get("messages")
    response = context[-1].content
    tool_messages = [message for message in context if message.type == "tool"]
    tool_context = tool_messages[-1].content
    return response,tool_context

def calculate_cv():
    input_value = input("请输入一个数字：")
    #输入的格式类似0.963/0.0052，前者是平均值，后者是方差，请计算变异系数
    # 请输入一个数字：0.963/0.0052
    # 变异系数为：0.0054
    average,variance = input_value.split("/")
    
    cv = math.sqrt(float(variance) )/float(average)
    print(cv)
    

if __name__ == "__main__":
    # os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    # graph = build_graph()
    # run_graph(graph,"你好")
    while True:
        calculate_cv()
