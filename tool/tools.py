import os

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

from config import config_init

config_init.init()
base_url = os.getenv('BASE_URL', "https://api.siliconflow.cn/v1")
api_key = os.getenv('API_KEY')

# 向量存储库(Qwen 编码器)
embeddings = OpenAIEmbeddings(model="Qwen/Qwen3-Embedding-8B", api_key=api_key, base_url=base_url)
VECTOR_STORE = InMemoryVectorStore(embeddings)


# 向量检索工具
def retrieve_context(query: str) -> str:
    """Retrieve information to help answer a query."""
    print(f"Retrieving {query}")
    retrieved_docs = VECTOR_STORE.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized


if __name__ == "__main__":
    arguments = {'query': "What is LangChain?"}
    tool_result = eval(f'retrieve_context(**{arguments})')
