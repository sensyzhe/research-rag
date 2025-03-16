from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # 更新导入
from langchain.tools.retriever import create_retriever_tool
import chromadb
import os

def get_retriever_tool(model_name= "moka-ai/m3e-base"):
    persist_directory = "./vector_db"
    cache_dir = "./model_cache"
    
    # 确保缓存目录存在
    os.makedirs(cache_dir, exist_ok=True)
    
    # 初始化嵌入模型
    print(f"初始化嵌入模型 {model_name}...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder=cache_dir,
        model_kwargs={'device': 'cpu'}
    )
    
    if os.path.exists(persist_directory):
        print("加载已存在的向量数据库...")
        client = chromadb.PersistentClient(path=persist_directory)
        vectorstore = Chroma(
            collection_name="rag-chroma",
            client=client,
            embedding_function=embedding_model
            
        )
    else:
        print("请先运行add_document.py文件，创建向量数据库...")
        return None
    retriever = vectorstore.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "牙科医学信息检索工具",
        "这是一个用于包含正畸学相关知识、牙科儿童相关信息、赛德阳光机构相关信息的工具。你可以通过这个工具检索医学信息",
    )

    print("向量数据库加载完成")
    return retriever_tool
