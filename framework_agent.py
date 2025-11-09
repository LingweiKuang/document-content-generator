import json
import os

import requests

from config import config_init
from entity.article import Chapter, Article

''' 构建文章行文框架 '''

config_init.init()
base_url = "https://api.siliconflow.cn/v1/chat/completions"
api_key = os.getenv('API_KEY')
model = "Qwen/Qwen3-30B-A3B-Thinking-2507"
# model = "Qwen/Qwen2.5-7B-Instruct"

system_prompt = """
您是中国科学院的杰出学者，拥有丰富的硕士论文写作经验。根据以下请求，请生成一份特定学术主题的硕士论文大纲。大纲应包括每章的整体内容，结构清晰，逻辑流畅，符合中国硕士论文的标准格式。

主题：<学术主题>

论文大纲示例如下：
```
{
  "摘要": "简要概述研究的目的、方法、主要发现和意义。摘要通常为300-500字。",
  "关键词": "列出3-5个关键词，反映论文的主要研究内容。",
  "第一章：引言": {
    "研究背景": "简要介绍研究的背景、目的和意义。",
    "文献综述": "总结国内外关于该主题的研究成果。",
    "研究问题与目标": "明确论文的研究问题、研究目标和研究范围。",
    "研究方法": "简要描述本研究采用的主要研究方法。",
    "论文结构": "概述每一章节的内容。"
  },
  "第二章：文献综述": {
    "研究领域的历史背景与发展趋势": "介绍该领域的主要国内外研究成果，概述不同学者的观点。",
    "当前研究的不足与本研究的创新点": "分析当前研究的不足，阐明本研究的创新之处。"
  },
  "第三章：研究方法与技术路线": {
    "研究设计": "介绍理论框架、假设及研究模型。",
    "数据收集": "描述实验设计、调查问卷、数据来源等。",
    "数据分析方法": "介绍用于数据分析的统计方法、工具或软件。",
    "技术路线": "呈现研究的技术路线图或流程图。"
  },
  "第四章：实证分析": {
    "数据分析": "详细描述分析过程，呈现主要结果。",
    "结果讨论": "将分析结果与假设或前人的研究进行比较，讨论差异。",
    "小结": "总结实证分析的主要发现。"
  },
  "第五章：结论与未来研究方向": {
    "主要结论": "总结研究发现，回答研究问题。",
    "研究贡献": "讨论本研究对该领域的贡献。",
    "研究局限与未来研究": "指出研究的局限性，并提出未来研究的可能方向。"
  },
  "参考文献": "按照标准的引文格式列出论文中引用的所有参考文献。",
  "附录": "包括必要的附加材料，如数据、调查问卷、代码等。"
}
```
"""

topic = "时序数据库查询处理逻辑错误检测技术"

messages = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": "主题思想: " + topic,
    }
]

payload = {
    "model": model,
    "messages": messages,
    "enable_thinking": True,
    "response_format": {"type": "json_object"},
}

headers = {
    "Authorization": "Bearer " + api_key,
    "Content-Type": "application/json"
}


def send_post_request():
    try:
        response = requests.post(base_url, headers=headers, json=payload)
        response.raise_for_status()  # 检查请求是否成功

        # 数据校验
        if len(response.json().get("choices")) != 1:
            print("response.json().get(choices) 长度不为1")

        return response.json().get("choices")[0].get("message").get("content")
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"状态码: {e.response.status_code}")
            print(f"错误信息: {e.response.text}")
        return None
    except Exception as e:
        print(f"请求失败: {e}")
        return None


def deserialization_article(json_res: str) -> Article:
    data = json.loads(json_res)
    paper = Chapter(title=topic, topic=topic)
    paper.deserialization(data)
    return paper


if __name__ == '__main__':
    try:
        json_res = send_post_request()
        print(json_res)
        article = deserialization_article(json_res)
        print(article)
    except Exception as e:
        print(e)
