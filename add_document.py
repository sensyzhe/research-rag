from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from tools import get_retriever,test_retriever_tool
import os
import sys
import shutil
if __name__ == "__main__":
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    model_name= "moka-ai/m3e-base"
    cache_dir = "./model_cache"
    persist_directory="./vector_db"
    local_store_path = "./docstore"
    # 初始化嵌入模型
    print(f"初始化嵌入模型 {model_name}...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder=cache_dir,
        model_kwargs={'device': 'cpu'}
    )

    print("开始加载文件...")
    retriever = get_retriever(embedding_model)


    # 从命令行获取文件路径
    if len(sys.argv) != 2:
        print("请提供文件路径")
        sys.exit(1)
    file_path = sys.argv[1]
    txt_files = ["data/"+file_path]
    docs = [TextLoader(file, encoding="utf-8").load() for file in txt_files]
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=10**9, chunk_overlap=0
        )
    print("开始分割文件...")
    doc_splits = text_splitter.split_documents(docs_list)
    retriever.add_documents(doc_splits)
    test_retriever_tool(embedding_model)
    #删除向量数据库
    # shutil.rmtree(persist_directory)
    # shutil.rmtree(local_store_path)
