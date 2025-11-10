import os

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from tool import tools_call, tools
from config import config_init

''' 博客 RAG 智能体 '''

config_init.init()
base_url = os.getenv('BASE_URL', "https://api.siliconflow.cn/v1")
api_key = os.getenv('API_KEY')
model = "Qwen/Qwen2.5-7B-Instruct"

if __name__ == '__main__':
    # 获取页面信息 - blog
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    assert len(docs) == 1
    print(f"Total characters: {len(docs[0].page_content)}")
    # blog 切片
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    all_splits = text_splitter.split_documents(docs)

    # blog 切片的向量存储
    document_ids = tools.VECTOR_STORE.add_documents(documents=all_splits)
    print(type(document_ids))

    tools = [tools.retrieve_context]

    system_prompt = (
        "You have access to a tool that retrieves context from a blog post. "
        "Use the tool to help answer user queries."
    )
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url
    )

    # 绑定 tool 的 agent
    agent = llm.bind_tools(tools)

    human_prompt = (
        "What is the **Memory stream**?\n\n"
        "Once you get the answer, look up common extensions of that method."
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": human_prompt}]

    res = tools_call.tool_call_invoke(agent, messages)
    print(res)
