from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tools import get_retriever, test_retriever_tool
from embedding_service import EmbeddingService
import os
import sys


if __name__ == "__main__":

    embedding_service = EmbeddingService()
    embedding_model = embedding_service.model
    retriever = get_retriever()

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
    
    test_retriever_tool()
    
