from langchain_openai import ChatOpenAI

# TODO 基于流程编排组合，完成多 Agent 协作，实现文档内容生成器

llm = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    # stream_usage=True,
    # temperature=None,
    # max_tokens=None,
    # timeout=None,
    # reasoning_effort="low",
    # max_retries=2,
    api_key="sk-gejifbjcqeltiruafqaozynoxshuqblmxfktsqzxaakjecpm",  # if you prefer to pass api key in directly instead of using env vars
    base_url="https://api.siliconflow.cn/v1",
    # organization="...",
    # other params...
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "what is your name?"),
]

ai_msg = llm.invoke(messages)
print(ai_msg)