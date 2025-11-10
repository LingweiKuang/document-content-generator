import json
import os

import requests

from config import config_init

# llm = ChatOpenAI(
#     model="Tongyi-Zhiwen/QwenLong-L1-32B",
#     # stream_usage=True,
#     # temperature=None,
#     # max_tokens=None,
#     # timeout=None,
#     # reasoning_effort="low",
#     # max_retries=2,
#     api_key=api_key,
#     # if you prefer to pass api key in directly instead of using env vars
#     base_url=base_url,
#     # organization="...",
#     # other params...
# )

config_init.init()
base_url = os.getenv('BASE_URL', "https://api.siliconflow.cn/v1")
api_key = os.getenv('API_KEY')
model = "Qwen/Qwen3-Next-80B-A3B-Thinking"

messages = [
    {"role": "system", "content": ""},
    {"role": "user",
     "content": "南方数码一家测绘地理信息行业的公司，集数据、软件、服务于一体，深耕测绘、自然资源、住建、智慧城市、农业农村、企业等领域。请仿造岳阳楼记写一篇：南方数码记"}
]
payload = {
    "model": model,
    "messages": messages,
    "stream": True,  # 启用流式处理
    "max_tokens": 16384,  # max_tokens必须小于等于16384
    "stop": ["null"],
    "temperature": 0.7,
    "top_p": 0.7,
    "top_k": 50,
    "frequency_penalty": 0.5,
    "n": 1,
    "response_format": {"type": "text"},
    # 注意：根据API文档，你可能需要移除或适当地填充tools字段
}

headers = {
    "Authorization": "Bearer " + api_key,
    "Content-Type": "application/json"
}

response = requests.post(base_url, json=payload, headers=headers, stream=True)

# 检查请求是否成功
if response.status_code == 200:
    first_reasoning_content_output = True
    first_content_output = True

    for chunk in response.iter_lines():
        if chunk:  # 过滤掉keep-alive新行
            chunk_str = chunk.decode('utf-8').strip()
            # print("==>",chunk_str)
            try:
                if chunk_str.startswith('data:'):
                    chunk_str = chunk_str[6:].strip()  # 去除"data:"前缀和之后的首位空格
                if chunk_str == "[DONE]":  # 完成了
                    print("\n\n============[DONE]============\n")
                    continue

                # 解析JSON
                chunk_json = json.loads(chunk_str)
                if 'choices' in chunk_json and isinstance(chunk_json['choices'], list) and len(
                        chunk_json['choices']) > 0:
                    choice = chunk_json['choices'][0]
                    delta = choice.get('delta', {})
                    # 获取思考过程信息
                    reasoning_content = delta.get('reasoning_content')
                    # 获取结果信息
                    content = delta.get('content')
                    # 获取完成原因
                    finish_reason = choice.get('finish_reason', None)
                    if finish_reason is not None:
                        print("\n\n\n==>查看结束原因，如果是stop，表示是正常结束的。finish_reason =", finish_reason)

                    # 打印思考过程：reasoning_content（如果有）
                    if reasoning_content is not None:
                        if first_reasoning_content_output:
                            print("思考过程:")
                            first_reasoning_content_output = False
                        print(reasoning_content, end='', flush=True)

                    # 打印结果内容：content（如果有）
                    if content is not None:
                        if first_content_output:
                            print("\n\n==============================\n结果:")
                            first_content_output = False
                        print(content, end='', flush=True)

            except json.JSONDecodeError as e:
                print(f"JSON解码错误: {e}", flush=True)
else:
    print(f"请求失败，状态码: {response.status_code}, 错误信息: {response.text}")
