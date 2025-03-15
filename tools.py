from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings  # 新的导入方式

from langchain.tools.retriever import create_retriever_tool
import os


def get_retriever_tool():
    print("开始加载向量数据库")
    txt_files = [
    "data/diseases_and_symptoms.txt",
    # "data/牙科儿童.txt",
        # "data/2.txt"
    ]

    # 加载所有txt文件内容
    docs = [TextLoader(file, encoding="utf-8").load() for file in txt_files]
    # docs = [TextLoader(file, encoding="utf-16").load() for file in txt_files]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)

    cache_dir = "./model_cache"
    os.makedirs(cache_dir, exist_ok=True)
    print("初始化嵌入模型")
    # 2. 初始化 embedding 模型时指定缓存目录
    embedding_model = HuggingFaceEmbeddings(
        model_name="moka-ai/m3e-base",
        cache_folder=cache_dir,
        model_kwargs={'device': 'cpu'}
    )
    print("初始化完成")
    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embedding_model,
    )
    retriever = vectorstore.as_retriever()


    retriever_tool = create_retriever_tool(
        retriever,
        "信息检索工具",
        "这是一个用于检索冷门疾病症状、牙科儿童相关信息的工具。你可以通过这个工具检索疾病和症状信息",
    )

    tools = [retriever_tool]
    print("向量数据库加载完成")
    return tools
