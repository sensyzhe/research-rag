from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # 更新导入
from langchain.tools.retriever import create_retriever_tool
import chromadb
from dotenv import load_dotenv
import os

def get_retriever_tool():
    persist_directory = "./vector_db"
    model_name = "moka-ai/m3e-base"
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
        print("创建新的向量数据库...")
        txt_files = ["data/diseases_and_symptoms.txt"]
        docs = [TextLoader(file, encoding="utf-8").load() for file in txt_files]
        docs_list = [item for sublist in docs for item in sublist]

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=100, chunk_overlap=50
        )
        doc_splits = text_splitter.split_documents(docs_list)
        
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=embedding_model,
            persist_directory=persist_directory
        )
    print(vectorstore.similarity_search("神秘人", k=2))
    retriever = vectorstore.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "信息检索工具",
        "这是一个用于检索冷门疾病症状、牙科儿童相关信息的工具。你可以通过这个工具检索疾病和症状信息",
    )

    print("向量数据库加载完成")
    return [retriever_tool]
