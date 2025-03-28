import dotenv
import os
from LLM import build_graph
import os
import dotenv
from ragas import EvaluationDataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from langchain_openai import ChatOpenAI


# dotenv.load_dotenv()
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
    print("开始构建图")
    graph = build_graph()
    print("图构建完成")
    # question = "论述正畸治疗中拔牙矫治的适应症和优缺点"
    # response,tool_context = get_response(question,graph)
    # print("response:",response)
    # print("tool_context:",tool_context)

    # sample_queries = ["关节的颞下颌韧带在什么情况下能发挥作用",
    #                   "关节盘在什么情况下会出现移位",
    #                   "准确咬合记录有哪些标准",
    #                   "种植钉植人稳定性受哪些因素影响"]
    # expected_responses = ["关节的颞下颌韧带只有在开口度超过20mm或更大时才发挥功能",
    #                       "1. 韧带被拉伸；2. 韧带撕裂；3. 韧带附着发生迁移。",
    #                       "1. 咬合记录一定不能引起牙齿移动或软组织的损伤；2. 咬合记录需要在口内确认准确度；3. 咬合记录在模型上就位要像口内一样准确；4. 需要在模型上确认咬合记录的准确度；5. 确保在保存或运送到技工室的过程中咬合记录没有变形。",
    #                       "种植钉的螺距（即螺纹间距） 紧密的螺距意味着螺纹间距离较小，宽松的螺距意味着螺纹间相距较远。骨质越致密，螺纹应越紧密。我们都知道，种植钉植人时的阻力大部分来自于骨皮质的接触，而较少来自骨髓质。由于在牙槽骨中骨皮质层较薄，种植钉的头部螺纹相对较密，从而可以提供与骨皮质间更多的接触，获得更大的穿透力以及良好的初期稳定性[28]。种植钉的长度　如果说与骨皮质的接触面积是决定种植钉稳定性的主要因素，那么种植钉与骨髓质的接触则显得没有那么重要，短种植钉可以同长种植钉一样发挥作用。但是，骨表面软组织的厚度也是一个重要的考量因素，伸入颧骨底部的骨钉需要更长才能触及骨皮质。贯穿牙槽骨直达另一侧的骨皮质的长种植钉，即双骨皮质钉确实获得了更大的稳定性[29]，但对大多数的应用来说，由于创伤较大，这样做并不值得。种植钉的直径　任何一颗植入牙槽突内的种植钉的直径都必须小于牙齿间的距离。目前，图 10-46 用作正畸支抗骨种植钉的种类，其中差异在于头部和骨荷的形状，种植钉形状和螺纹形状（形式），螺纹之间的距离（间距）。每颗螺钉都需要一个特殊的手柄（推进器），和螺钉头部的基板相符合，种植钉上钢丝或弹簧结扎的方法各不相同。正畸应用中的骨种植钉的最佳特性主要取决于其植入位置和所承受的力的大小，其次取决于它的难易程度或易用性。",
    #                       ]

    # dataset = []

    # for similarity_threshold in [0.7,0.75,0.8,0.85,0.9]:
    #     graph = build_graph(similarity_threshold)
    # for query,reference in zip(sample_queries,expected_responses):

    #     response,relevant_docs = get_response(query,graph)
        
    #     dataset.append(
    #         {
    #             "user_input":query,
    #             "retrieved_contexts":relevant_docs,
    #             "response":response,
    #             "reference":reference
    #         }
    #     )
    #     evaluation_dataset = EvaluationDataset.from_list(dataset)
    #     metrics = [LLMContextRecall(), Faithfulness(), FactualCorrectness()]
        
    #     result = evaluate(dataset=evaluation_dataset,metrics=metrics)
    #     print(similarity_threshold,":",result)