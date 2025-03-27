from langchain_huggingface import HuggingFaceEmbeddings
import os

class EmbeddingService:
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if EmbeddingService._model is None:
            # 设置代理
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            cache_dir = "./model_cache"
            os.makedirs(cache_dir, exist_ok=True)
            
            print("正在初始化嵌入模型...")
            EmbeddingService._model = HuggingFaceEmbeddings(
                model_name="moka-ai/m3e-base",
                cache_folder=cache_dir,
                model_kwargs={'device': 'cpu',},
            )
            print("嵌入模型初始化完成")
    
    @property
    def model(self):
        return EmbeddingService._model

# 使用示例
if __name__ == "__main__":
    # 第一次获取实例，会初始化模型
    service1 = EmbeddingService()
    
    # 第二次获取实例，会复用已初始化的模型
    service2 = EmbeddingService()
    
    # 使用模型
    texts = ["这是一个测试文本"]
    embeddings = service1.model.embed_documents(texts)
    print(embeddings) 