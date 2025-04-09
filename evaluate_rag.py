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
                      "关节盘在什么情况下会出现移位",
                      "准确咬合记录有哪些标准",
                      "垂直向发育不足的患儿早期有什么症状",
                      "当儿童被诊断为髁突骨折时,让儿童正常生长的关键是什么",
                      "持续性压力作用于牙齿会引发一系列什么变化",
                      "在正畸临床中产生单力偶系统需满足什么条件"]
    expected_responses = ["关节的颞下颌韧带只有在开口度超过20mm或更大时才发挥功能",
                          "1. 韧带被拉伸；2. 韧带撕裂；3. 韧带附着发生迁移。",
                          "1. 咬合记录一定不能引起牙齿移动或软组织的损伤；2. 咬合记录需要在口内确认准确度；3. 咬合记录在模型上就位要像口内一样准确；4. 需要在模型上确认咬合记录的准确度；5. 确保在保存或运送到技工室的过程中咬合记录没有变形。",
                          "他们常表现出下颌平面角小（骨性深覆验），下颌开支长的倾向。下颌骨常向前生长，具有向上、向前旋转的趋势。",
                          "维持口颌功能",
                          "1. 牙周组织液未压缩，牙槽骨弯曲，压电信号产生。2. 牙周组织液排出，牙齿在牙周膜间隙内移动。3. 牙周膜压力侧部分血管受压，牵张侧膨胀；牙周纤维细胞机械扭曲形变。4. 血流改变，氧分压开始改变，前列腺素和细胞因子释放。5. 代谢改变，化学信使影响细胞活性，酶水平改变。6. cAMP 水平升高，牙周膜内细胞开始分化。7. 破骨细胞／成骨细胞改建牙槽窝，牙齿开始移动。8. 牙周膜压力侧血管阻塞。9. 牙周膜压力侧血流阻断。10. 牙周膜压力侧细胞死亡。11. 邻近的骨髓腔内细胞分化，开始潜掘性吸收。12. 7～14d 后，滑掘性吸收，清除邻近受压区域的硬骨板，牙齿开始移动。",
                          "①悬臂弹簧或辅弓的一端被放入一颗或多颗（增强支抗时使用）位于稳定区段的牙齿的托槽或频管中；②另一端以单点施力的方式结扎于一颗或多颗需要被移动的牙齿上"]

    
    dotenv.load_dotenv()
    llm = ChatOpenAI(
        model="deepseek-v3",
        openai_api_key="sk-5e537d1de9a84175bd6c486c284e57a5",
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    

    
    logging.info("开始评估过程")
    thread_id = 0
    for similarity_threshold in range(50,92,3):
        similarity_threshold = similarity_threshold / 100
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
        
    logging.info("评估过程完成")