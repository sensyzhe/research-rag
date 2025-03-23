from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # 更新导入
from langchain.tools.retriever import create_retriever_tool
import chromadb
import os
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore

def get_retriever( model_name= "moka-ai/m3e-base",embedding_model=None):
    persist_directory = "./vector_db"
    cache_dir = "./model_cache"
    local_store_path = "./docstore"
    # 设置代理
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    # 确保缓存目录存在
    os.makedirs(cache_dir, exist_ok=True)
    
    # 初始化嵌入模型
    if embedding_model is None:
        print(f"初始化嵌入模型 {model_name}...")
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=cache_dir,
            model_kwargs={'device': 'cpu',},
        )
        print("初始化嵌入模型完成")
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=8)

    # Assuming `splits` is your list of documents
    client = chromadb.PersistentClient(path=persist_directory)
    vectorstore = Chroma(
        collection_name="rag-chroma",
        client=client,
        embedding_function=embedding_model 
    )

    # The storage layer for the parent documents
    
    local_store = LocalFileStore(local_store_path)
    docstore = create_kv_docstore(local_store)
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    return retriever

def get_retriever_tool(model_name= "moka-ai/m3e-base",embedding_model=None):
    persist_directory = "./vector_db"
    # 设置代理
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    if os.path.exists(persist_directory):
        retriever = get_retriever(model_name=model_name,embedding_model=embedding_model)
    else:
        print("请先运行add_document.py文件，创建向量数据库...")
        raise ValueError("向量数据库不存在")
    
    retriever_tool = create_retriever_tool(
        retriever,
        "信息检索工具",
        """这是一个用于检索包含正畸学、功能牙合学、儿童牙科诊疗行为管理相关信息、赛德阳光机构相关信息的工具。
        你可以通过这个工具检索医学信息,请注意你应该利用工具返回的信息针对用户问题给出自己的回答，不要直接返回工具返回的信息""",
    )

    print("向量数据库加载完成")
    return retriever_tool

def test_retriever_tool(model_name= "moka-ai/m3e-base",embedding_model=None):
    retriever_tool = get_retriever_tool(model_name,embedding_model)
    while True:
        user_input = input("请输入要检索的信息：")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        print(retriever_tool.invoke(user_input))

if __name__ == "__main__":
    test_retriever_tool()
