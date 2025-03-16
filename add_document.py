from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # 更新导入
from langchain.tools.retriever import create_retriever_tool
import chromadb
import os
import sys

if __name__ == "__main__":
    # 从命令行获取文件路径
    if len(sys.argv) != 2:
        print("请提供文件路径")
        sys.exit(1)
    file_path = sys.argv[1]
    
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
    
    client = chromadb.PersistentClient(path=persist_directory)
    vectorstore = Chroma(
        collection_name="rag-chroma",
        client=client,
        embedding_function=embedding_model
        
    )
    print("开始加载文件...")
    txt_files = ["data/"+file_path]
    docs = [TextLoader(file, encoding="utf-8").load() for file in txt_files]
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500, chunk_overlap=100
        )
    print("开始分割文件...")
    doc_splits = text_splitter.split_documents(docs_list)
    print("执行嵌入模型...")
        # 增量更新向量数据库
    vectorstore.add_documents(doc_splits)