from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.tools.retriever import create_retriever_tool
import chromadb
import os
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.retrievers.document_compressors import DocumentCompressorPipeline,EmbeddingsFilter
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_text_splitters import CharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from embedding_service import EmbeddingService

persist_directory = "./vector_db"
local_store_path = "./docstore"


def get_retriever():
    
    # 使用嵌入服务
    embedding_service = EmbeddingService()
    embedding_model = embedding_service.model
    
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)

    client = chromadb.PersistentClient(path=persist_directory)
    vectorstore = Chroma(
        collection_name="rag-chroma",
        client=client,
        embedding_function=embedding_model 
    )

    local_store = LocalFileStore(local_store_path)
    docstore = create_kv_docstore(local_store)
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    return retriever

def get_retriever_tool(similarity_threshold=0.75):
    
    if os.path.exists(persist_directory):
        retriever = get_retriever()
    else:
        print("请先运行add_document.py文件，创建向量数据库...")
        raise ValueError("向量数据库不存在")
    
    embedding_service = EmbeddingService()
    embedding_model = embedding_service.model
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=10,
                                             separators=["。", "？", "！", "\n"])
    
    #一系列压缩工具
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embedding_model)
    relevant_filter = EmbeddingsFilter(embeddings=embedding_model, similarity_threshold=similarity_threshold)
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter]
    )

    # 检索工具组装
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=retriever
    )
    retriever_tool = create_retriever_tool(
        compression_retriever,
        "信息检索工具",
        """这是一个用于检索包含正畸学、功能牙合学、儿童牙科诊疗行为管理相关信息、赛德阳光机构相关信息的工具。
        你可以通过这个工具检索医学信息,请注意你应该利用工具返回的信息针对用户问题给出自己的回答，不要直接返回工具返回的信息""",
    )

    print("向量数据库加载完成")
    return retriever_tool

def test_retriever_tool():
    while True:
        similarity_threshold = float(input("请输入相似度阈值："))
        user_input = input("请输入要检索的信息：")
        retriever_tool = get_retriever_tool(similarity_threshold)
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        print(retriever_tool.invoke(user_input))

if __name__ == "__main__":
    test_retriever_tool()
    # graph = build_graph()