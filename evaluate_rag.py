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
import numpy as np
import math

response = """在牙科中，**对刃颌（Edge-to-Edge Bite）的稳定性**主要取决于以下关键因素，结合临床咬合学原则和文本中的治疗目标可总结如下：

---

### **1. 正中关系位的稳定性**
- **均匀接触**：在可验证的正中关系位（CR），所有牙齿需均匀接触（文本目标1）。若对刃颌在CR位无稳定、分散的接触点，可能导致咬合不稳定或异常负荷。
- **牙尖交错关系**：对刃颌缺乏正常的覆𬌗覆盖时，需通过修复或正畸重建牙尖交错关系以增强稳定性。

---

### **2. 前导功能的协调性**
- **前导与下颌运动的匹配**：前牙引导（前导）必须与患者的髁突运动（髁导）协调（文本目标2）。对刃颌若前导缺失或不足（如前牙切端平齐），侧方运动时后牙可能无法有效分离，导致干扰和进一步损耗。
- **即刻后牙分离**：下颌非正中运动时，后牙需迅速脱离接触（文本目标3）。对刃颌若无法实现这一点，可能引发磨耗或关节问题。

---

### **3. 咬合垂直距离（VDO）的维持**
- **牙槽骨代偿**：严重损耗时，牙槽骨可能通过伸长代偿釉质丧失（文本末段），但需评估VDO是否因对刃颌而异常。过度丧失VDO可能影响颞下颌关节和肌肉功能。
- **修复设计**：若需修复，需谨慎选择是否增加VDO（文本治疗选择2），避免因不当调整加重不稳定。

---

### **4. 后牙咬合的协调性**
- **前导与髁导的平衡**：后牙咬合需在前导和髁导之间达到"长正中"协调（文本后牙修复原则）。对刃颌若后牙存在早接触或干扰，会破坏稳定性。
- **侧方干扰的消除**：即使对刃颌看似"平坦"，仍需排查侧方运动中的微小干扰（文本现象2）。

---

### **5. 副功能运动的控制**
- **夜磨牙或紧咬牙习惯**：对刃颌患者若存在副功能运动，需使用𬌗垫（文本目标6）以减少异常负荷。缺乏前导缓冲时，磨牙症可能加速牙齿损耗。

---

### **6. 牙列与中性区的关系**
- **正畸适应性**：牙齿位置需与舌、唇肌力平衡的"中性区"一致（文本正畸部分）。对刃颌若因牙槽骨排列异常导致，可能需正畸或正颌手术（文本牙槽骨重排问题）。

---

### **7. 修复体的设计**
- **美学与功能的权衡**：前牙区对刃颌修复需兼顾美观和功能引导（文本治疗选择3）。例如，通过全冠恢复适当覆𬌗覆盖。

---

### **总结**
对刃颌的稳定性核心在于：  
**① 正中关系位的均匀接触；  
② 前导与髁导的协调；  
③ 后牙在非正中运动中的即刻分离；  
④ 咬合垂直距离与牙槽骨代偿的平衡；  
⑤ 消除副功能运动的干扰。**  
若无法通过调𬌗实现，需结合修复、正畸或𬌗垫治疗。
"""

relevant_docs = """损耗问题的治疗计划
过度损耗问题的治疗应达到以下6个目标：
1. 在可验证的正中关系位所有牙齿均匀接触。
2. 前导与患者的正常下颌功能运动协调一致。
3. 当下颌从正中关系位向任何方向运动时，后牙立刻脱离咬合接触。
4. 对损耗超过釉质层的牙齿进行修复

。

咬合损耗的修复时机
并非所有的咬合损耗都需要修复。即使损耗到达牙本质也可能并不需要治疗。如果调殆可以消除损耗的原因，使损耗表面不 
再发生副功能接触，暴露的牙本质可能会完整地保存多年

。
第三十五章 解决咬合损耗问题

图35-1 A. 严重损耗最初表现为化学破坏的形式。咬合关系分析表明，下颌非正中运动可到达所有的损耗区域。B. 患者没有
使用腐蚀性或研磨类物质的习惯。诊断为某种形式的釉质发育不全，使得牙齿对磨耗特别敏感。全覆盖修复体具有较成功的 
远期效果

。
5. 与患者沟通，以便于理解正常的下颌位置除了吞咽时牙齿都是分开的。建议："嘴唇接触，牙齿分开"。
6. 调整咬合后还存在夜磨牙习惯者，使用夜磨牙防护器。

决定采取哪种方法治疗损耗直接取决于牙列需要接受哪种改变后可以达到前4个目标。将损耗与前导髁导结合相关联        

。然后应该直接针对牙列的改变制订治疗计划，以达到完全协调一致，对下颌的任何功能性位置或非正中运动都没有干扰。 
通常损耗方式本身就是确定下颌功能运动轨迹的关键。

分析任何牙列时，我们都应该区分生理性损耗和过度损耗。生理性损耗是正常的

对所有严重损耗患者的治疗目标都是：在下颌离开正中关系位时，能够在完美的前导引导下使后牙咬合分离。

验别损耗的原因

最终修复体就位。
第三部分 治疗

修复严重损耗的后牙
后牙咬合的修复，取决于首先确定正确的前导。对各种不同类型咬合损耗的分析，应该首先聚焦于前牙。后牙必须在前导和 
髁导之间契合，但前导和髁导不能相互干扰。因此，建立稳定的接触关系是后牙咬合的主要目标

。损耗在上前牙舌面从各个方向延伸到龈缘，下前牙由于舌的保护而免受严重损伤。

严重损耗的牙列是牙科学中最大的挑战之一。但如果按照序列化制订治疗计划的原则，从确定协调稳定完全就位的颞下颌关 
节开始，严格遵守正确治疗顺序，就能简化严重损耗的治疗计划

。

在讨论具体的治疗计划前，我们要先理解以下6种可以观察到的现象：
1. 严重损耗不会导致咬合垂直距离的丧失。
2. 严重损耗不是消除所有侧方2干扰（即使面看似已被磨平）。
3. 只有在下颌正常功能或副功能运动中，上颌牙齿阻碍了下颌牙齿，才会发生严重的磨耗

。因为成功解决大部分损耗问题要求除了正中关系位，其他所有颌位后牙分离咬合，故而在分析严重损耗问题时必须将焦点 
集中在如何最好地实现这个目标。因为后牙的咬合分离受前导和髁导的共同影响，因此必须对这两者进行分析。只要出现广 
泛性严重损耗，通常就可以见到前导和/或髁导严重变形

。
长期损耗有时候可导致后牙损耗接近牙龈，对这种严重损耗的牙齿通常有4种治疗选择：
1. 钉洞固位的纯金修复体。在暴露的牙本质上通过平行的钉洞固位进行修复，不显著增加咬合垂直距离。但因为美学方面的
问题，这种方法在前牙区不一定能被接受。
2. 增加咬合垂直距离

。过度损耗会导致釉面出现不可接受的破坏，并破坏前牙结构从而影响前导功能或美观。过度损耗是有诊断特征的。它与直 
接干扰下颌功能运动和副功能运动的牙面相关。如果牙体结构没有阻碍下颌运动，就不会发生过度损耗。对下颌运动的直接 
干扰点或者在滑动的终点都可能会激发过度损耗

。否则就会增加咬合垂直距离，可能会真正加重某些患者的问题。在分析严重损耗的问题时，应该考虑到如果没有水平和垂 
直向的釉干扰，正常的肌肉功能是如何移动下颌的。换句话说，应该对下颌功能进行分析并判定牙列的各个部分是为何以及 
如何干扰下颌运动的

。
3. 是否要求修复体满足美学需求？损耗的牙齿可能不美观。修复体如果和损耗的邻牙匹配的话，就严重损害了本来可以达到
的美学效果。应该告知患者各种治疗方法。
4. 需要相对明确最终是否需要修复体？如果是，应告知患者可能的治疗时间范围，如果患者选择延迟治疗则应该定期检查患
者的状况

。如果对如何制订整个治疗计划的前牙区段尚存疑虑的话，请复习第十五～第十八章关于治疗流程的描述。然而，还要根据 
临床检查架上模型的研究解决一些基础问题

。

牙-牙槽骨区段重新排列
·需要矫正的程度是否过于严重以至于不能采用简单的正畸或联合治疗？
第三十章　咬合疾患的序列化治疗计划

·外科的方法是否更有优势？
·是否能采取正颌外科的方法？

牙槽骨位置重排
·牙槽骨位置是否有问题？
·确定那些有错误的区段

。
·如果解除后牙干扰后，前牙能否接触？
·前牙是否需要稳定止动？
·下前牙切端是否发生任何磨耗问题？

调磨改形（调）
·该治疗能解决问题吗？
·该治疗能使前牙在正中关系位发生接触吗？
·该治疗是否需要磨损好的牙齿？
·该治疗能否部分有助于达到预期的结果？
如有需要，在模型上进行诊断性调磨

。是否需要修复损耗面取决于下面几个问题的答案：
1. 延迟对损耗的修复是否会使治疗更加复杂？通常情况下即使发生进一步的损耗，修复方式可能还是不变的。这种情况不需
要紧急处理，特别是如果有机会通过调整咬合来纠正损耗问题时。
2. 必须通过修复来控制牙齿敏感吗？牙本质暴露在不同患者身上会有不同的反应

。根据临床经验判断不暴露牙本质前提下的调磨。

重新排牙（正畸）
·常规正畸重排牙齿能否达到保持稳定的咬合接触？（观察中性区）
·牙齿改形和牙齿位置重排结合是否能达到更好的效果？
如有需要，在模型上模拟牙齿移动以评估正畸的可能性。咨询正畸医生

在一生中都会持续这种适应过程。如果机械系统各部件都正确地相互关联，这种适应性是非常有利的。但如果系统相互关联 
部分的功能不协调，这种适应性可能会加剧牙列的破坏。

由于牙槽骨的伸长弥补了釉面损耗量，严重损耗牙齿的修复就不是采用简单地恢复丧失牙体组织的方式
"""




# dotenv.load_dotenv()
def get_response(question,graph,thread_id="1"):
    config = {"configurable": {"thread_id": thread_id}}
    
    context =   graph.invoke({"messages": [{"role": "user", "content": question}]},config).get("messages")
    response = context[-1].content
    tool_messages = [message for message in context if message.type == "tool"]
    tool_context = tool_messages[-1].content
    return response,tool_context

def evaluate_rag(question, expected_response, model_name, times=10):
    # 配置日志
    log_filename = f'{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        filename=log_filename,
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
    llm = ChatOpenAI(
        model=model_name,
        openai_api_key="sk-5e537d1de9a84175bd6c486c284e57a5",
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    logging.info("开始评估过程")
    LLMContextRecall_list = []
    Faithfulness_list = []
    FactualCorrectness_list = []
    for i in range(times):
        
        dataset = []
        for query,reference in zip(sample_queries,expected_responses):
            
            
            dataset.append(
                {
                    "user_input":query,
                    "retrieved_contexts": [relevant_docs],
                    "response":response,
                    "reference":reference
                }
            )
        evaluation_dataset = EvaluationDataset.from_list(dataset)
        #context recall:指
        metrics = [LLMContextRecall(), Faithfulness(), FactualCorrectness()]

            
        print("开始评估")
        result = evaluate(dataset=evaluation_dataset,metrics=metrics,llm=LangchainLLMWrapper(llm))
        logging.info(f" {model_name} 的评估结果: {result}")

        if not math.isnan(result["context_recall"][0]):
            print(type(result["context_recall"][0]))
            # 确保 context_recall 是数值类型
            context_recall_value = (result["context_recall"][0])
            LLMContextRecall_list.append(context_recall_value)
        if not math.isnan(result["faithfulness"][0]):
            # 确保 faithfulness 是数值类型
            print(type(result["faithfulness"][0]))
            faithfulness_value =(result["faithfulness"][0])
            Faithfulness_list.append(faithfulness_value)
        if not math.isnan(result["factual_correctness(mode=f1)"][0]):
                # 确保 factual_correctness 是数值类型
            print(type(result["factual_correctness(mode=f1)"][0]))
            factual_correctness_value = (result["factual_correctness(mode=f1)"][0])
            FactualCorrectness_list.append(factual_correctness_value)

    mean_list = [sum(LLMContextRecall_list)/len(LLMContextRecall_list),sum(Faithfulness_list)/len(Faithfulness_list),sum(FactualCorrectness_list)/len(FactualCorrectness_list)]
    logging.info("平均分：LLMContextRecall:{},Faithfulness:{},FactualCorrectness:{}".format(mean_list[0],mean_list[1],mean_list[2]))
    # 计算方差
    var_list = [np.var(LLMContextRecall_list),np.var(Faithfulness_list),np.var(FactualCorrectness_list)]
    logging.info("方差：LLMContextRecall:{},Faithfulness:{},FactualCorrectness:{}".format(var_list[0],var_list[1],var_list[2]))
    # 计算变异系数,如果平均值为0，则变异系数为0
    cv_list = [np.std(LLMContextRecall_list)/mean_list[0] if mean_list[0] != 0 else 0,
               np.std(Faithfulness_list)/mean_list[1] if mean_list[1] != 0 else 0,
               np.std(FactualCorrectness_list)/mean_list[2] if mean_list[2] != 0 else 0]
    logging.info("变异系数：LLMContextRecall:{},Faithfulness:{},FactualCorrectness:{}".format(cv_list[0],cv_list[1],cv_list[2]))
    logging.info("评估过程完成")


if __name__ == "__main__":
    sample_queries = [
                      "牙科中对刃翰是否稳定主要取决于什么因素？"
                      ]
    expected_responses = [
                          """对刃翰是否稳定主要取决于两方面的因素：
                                1. 与中性区的相互协调；
                                2. 下颌功能运动范围内没有殆干扰。
                            """
                          ]

    
    dotenv.load_dotenv()
    # 从命令行获取模型名称
    
    
    
    
    model_names =["qwen-turbo","qwen-max","qwen-long","llama-4-maverick-17b-128e-instruct"]
    for model_name in model_names:
        evaluate_rag(sample_queries, expected_responses, model_name, 1)
        # Reset logging configuration to avoid duplicate handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
    
    