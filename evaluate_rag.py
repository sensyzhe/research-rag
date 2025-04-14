from LLM import build_graph
import os
import dotenv
from ragas import EvaluationDataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from langchain_openai import ChatOpenAI
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    filename=f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    encoding='utf-8'
)

# 同时输出到控制台和文件
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

# 设置 HTTP 请求日志级别为 WARNING，这样就不会显示 INFO 级别的请求日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# dotenv.load_dotenv()
def get_response(question,graph,thread_id="1"):
    config = {"configurable": {"thread_id": thread_id}}
    
    context =   graph.invoke({"messages": [{"role": "user", "content": question}]},config).get("messages")
    response = context[-1].content
    tool_messages = [message for message in context if message.type == "tool"]
    tool_context = tool_messages[-1].content
    return response,tool_context

if __name__ == "__main__":
    sample_queries = ["关节的颞下颌韧带在什么情况下能发挥作用",
                      ]
    expected_responses = ["关节的颞下颌韧带只有在开口度超过20mm或更大时才发挥功能",
                          ]

    
    dotenv.load_dotenv()
    llm = ChatOpenAI(
        model="deepseek-v3",
        openai_api_key="sk-5e537d1de9a84175bd6c486c284e57a5",
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    

    
    logging.info("开始评估过程")
    thread_id = 0
    similarity_threshold=0.75
    LLMContextRecall_list = []
    Faithfulness_list = []
    FactualCorrectness_list = []
    for i in range(5):
        graph = build_graph(similarity_threshold)
        dataset = []
        thread_id += 1
        for query,reference in zip(sample_queries,expected_responses):

            response,relevant_docs = get_response(query,graph,str(thread_id))
            
            relevant_docs = [relevant_docs]
            dataset.append(
                {
                    "user_input":query,
                    "retrieved_contexts":relevant_docs,
                    "response":response,
                    "reference":reference
                }
            )
        evaluation_dataset = EvaluationDataset.from_list(dataset)
        #context recall:指
        metrics = [LLMContextRecall(), Faithfulness(), FactualCorrectness()]

            
        print("开始评估")
        result = evaluate(dataset=evaluation_dataset,metrics=metrics,llm=LangchainLLMWrapper(llm))
        logging.info(f"相似度阈值 {similarity_threshold} 的评估结果: {result}")

        if result["LLMContextRecall"] is not None:  
            LLMContextRecall_list.append(result["LLMContextRecall"])
        if result["Faithfulness"] is not None:
            Faithfulness_list.append(result["Faithfulness"])
        if result["FactualCorrectness"] is not None:
            FactualCorrectness_list.append(result["FactualCorrectness"])

    logging.info("平均分：LLMContextRecall:{},Faithfulness:{},FactualCorrectness:{}".format(sum(LLMContextRecall_list)/len(LLMContextRecall_list),sum(Faithfulness_list)/len(Faithfulness_list),sum(FactualCorrectness_list)/len(FactualCorrectness_list)))
    logging.info("评估过程完成")