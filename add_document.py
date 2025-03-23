from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from tools import get_retriever,test_retriever_tool
import os
import sys
import shutil

if __name__ == "__main__":
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    model_name = "moka-ai/m3e-base"
    cache_dir = "./model_cache"
    persist_directory = "./vector_db"
    local_store_path = "./docstore"

    # 初始化嵌入模型
    print(f"初始化嵌入模型 {model_name}...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder=cache_dir,
        model_kwargs={'device': 'cpu'}
    )

    print("开始加载文件...")
    retriever = get_retriever(model_name=model_name,embedding_model=embedding_model)

    # 从命令行获取文件路径
    if len(sys.argv) < 2:
        print("使用方法: python add_document.py file1.txt file2.txt ...")
        sys.exit(1)

    # 获取所有文件路径
    file_paths = sys.argv[1:]
    txt_files = ["data/" + file_path for file_path in file_paths]
    
    # 检查文件是否存在
    for file_path in txt_files:
        if not os.path.exists(file_path):
            print(f"错误：文件 {file_path} 不存在")
            sys.exit(1)

    print(f"正在处理 {len(txt_files)} 个文件...")
    
    # 加载所有文件
    docs = []
    for file in txt_files:
        try:
            print(f"正在加载文件: {file}")
            doc = TextLoader(file, encoding="utf-8").load()
            docs.extend(doc)
        except Exception as e:
            print(f"加载文件 {file} 时出错: {str(e)}")
            continue

    if not docs:
        print("没有成功加载任何文件")
        sys.exit(1)

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=10**9, chunk_overlap=0
    )
    
    print("开始分割文件...")
    doc_splits = text_splitter.split_documents(docs)
    print(f"文件分割完成，共 {len(doc_splits)} 个文档片段")
    
    print("开始添加文档到向量数据库...")
    retriever.add_documents(doc_splits)
    print("文档添加完成")
    
    test_retriever_tool(model_name,embedding_model)
    
    #删除向量数据库
    # shutil.rmtree(persist_directory)
    # shutil.rmtree(local_store_path) 