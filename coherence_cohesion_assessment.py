import json
import os
from json import JSONDecodeError
from typing import List, Tuple, Union

from langchain_openai import ChatOpenAI

import framework_agent
from chapter_writing_agent import writing_content
from config import config_init
from entity import evaluation_indicator
from entity.article import Chapter, Article, Content

''' 文本流畅度评分 '''
# TODO 感觉大模型评分不是很可靠，使用缓存，基于概率时，是否会导致同一评分出现的频率大大增加

config_init.init()
base_url = os.getenv('BASE_URL', "https://api.siliconflow.cn/v1")
api_key = os.getenv('API_KEY')
# model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
model = "Qwen/Qwen2.5-7B-Instruct"

llm = ChatOpenAI(
    model=model,
    api_key=api_key,
    base_url=base_url,
)

json_llm = llm.bind(response_format={"type": "json_object"})

system_prompt = """
你是一位中国中科院院士，擅长对硕士学术论文进行修订，并判定段落内容的衔接度。你的目标是根据给定论文段落，从多个角度对 **{原文}** 进行评估，判定语句间、段落间的衔接是否流畅。请基于以下**评分标准**对段落进行评分，并给出 **{评语(JSON)}**：

评分标准：
1. **语句用词专业度**：评估段落中的术语、表达是否符合学术标准，语言是否准确、专业。
2. **段落逻辑结构**：评估段落内句与句之间、段落与段落之间的逻辑关系是否清晰，是否自然地衔接。
3. **过渡语句的使用**：检查是否有恰当的过渡语句，确保内容间的顺畅过渡，避免突兀的跳跃。
4. **语言流畅度**：评估段落是否易读，语句是否通顺，避免不必要的重复或语法错误。
5. **内容重复度**：评估段落中是否存在重复的观点、措辞或信息，确保内容简洁且不重复。

示例：
** 文章主题 **：基于机器学习的图像分类算法优化\n摘要\n

** 原文 **：
```随着图像分类领域的快速发展，深度学习技术，尤其是卷积神经网络（CNN），推动了该领域的进步。现代图像分类算法在许多基准数据集（如ImageNet）上已经达到了或超过了人类水平，广泛应用于安全监控、自动驾驶、医学图像分析和内容推荐等领域。预训练模型（如ResNet、VGG、Transformer变种等）的迁移学习能力使得即使在小规模数据集上也能实现高性能，显著加速了实际应用的部署。```

** 评语(JSON) **：
```
[{"评估维度":"语句用词专业度","评分":7.5/10,"详细评价":"措辞方面，句一“推动了该领域的进步”过于抽象和通俗化，未达到学术论文对高密度信息表达的要求。此外，对“超过了人类水平”这一核心成就的论断，应加入“ImageNet等特定大规模视觉基准”的限定语，以确保学术严谨性。"},{"评估维度":"段落逻辑结构","评分":8.5/10,"详细评价":"宏观结构良好，但句二和句三之间存在逻辑跳跃。段落直接从高性能带来的“广泛应用”跳至“迁移学习”解决方案，缺少对应用中遇到的关键部署挑战（如数据稀疏、泛化性不足）的明确引入。"},{"评估维度":"过渡语句的使用","评分":7/10,"详细评价":"句间衔接相对生硬，未能使用强引导性的过渡词，来明确指示信息焦点从“应用成果”向“解决方案”的转移。"},{"评估维度":"语言流畅度","评分":8/10,"详细评价":"语言流畅，但由三个复杂长句构成，阅读节奏变化不足。句二中应用领域列表过长，使得句子承载的信息负荷较高，可能影响可读性。"},{"评估维度":"内容重复度","评分":8/10,"详细评价":"表面无重复，但存在低密度信息冗余。句一“推动了该领域的进步”是抽象描述，其信息价值已被句二的具象成果“达到了或超过了人类水平”完全涵盖和超越。"}]
```
"""


# TODO 判断评语是否集中在指定的原文内容上，而不进行衍生
def text_scoring(messages: List[Tuple[str, str]], iteration_num) -> List[evaluation_indicator.Indicator]:
    if iteration_num >= 3:
        print(f"错误: 重试次数超出!")
        return []

    try:
        ai_msg = json_llm.invoke(messages)
        evaluation_res = ai_msg.content
        data = json.loads(evaluation_res)

        # 将评语反序列化为对象
        indicators = []
        for res in data:
            mapped_data = {evaluation_indicator.field_mapping[key]: value for key, value in res.items()}
            indicator = evaluation_indicator.Indicator(**mapped_data)
            indicators.append(indicator)
        return indicators
    except JSONDecodeError as e:
        print(f"模型返回 JSON 格式错误: {e}")
        return text_scoring(messages, iteration_num + 1)
    except Exception as e:
        print(f"error: {e}")
        return text_scoring(messages, iteration_num + 1)


def article_score(article: Article, prefix_prompt: str) -> List[Union[Chapter, Article]]:
    prefix_prompt += article.to_prompt() + "\n"
    # 对子节点进行评分
    updated_sections = []
    for section in article.subsections:
        if isinstance(section, Chapter):
            # 递归评分章节
            chapter = Chapter(title=section.summary_key())
            chapter.subsections = article_score(section, prefix_prompt)
            updated_sections.append(chapter)
        else:
            # 遍历 article 进行评分 => 按照框架节点进行打分, 主题-章节-段落中心思想(可能包含多段内容)
            content_prompt = "** 文章主题 **:{}{}\n \n** 原文 **：```{}```\n** 评语(JSON) **：".format(
                prefix_prompt, section.summary_key(), section.to_prompt())
            messages = [("system", system_prompt), ("human", content_prompt)]
            # 内容可由大模型修订则执行评分
            if section.isModifiable:
                indicators = text_scoring(messages, 0)
                remark = json.dumps([indicator.model_dump() for indicator in indicators], ensure_ascii=False)
            else:
                remark = ""
            content = Content(summary=section.summary_key(), text=section.get_text(),
                              isModifiable=section.isModifiable, remark=remark)
            updated_sections.append(content)

    return updated_sections


if __name__ == '__main__':
    # 引用框架
    article_framework = None
    try:
        json_res = """
            {
              "摘要": "时序数据库在物联网、金融交易、智能监控等领域具有广泛应用，而查询处理中的逻辑错误会导致数据不一致、业务中断等问题。本研究针对时序数据库查询处理逻辑错误检测技术展开系统性研究，提出基于语义分析与特征学习的错误检测模型。研究首先分析了时序查询逻辑错误的类型与特征，构建了包含时序语义约束模型；其次，设计了基于图神经网络的错误检测算法，实现了对复杂查询逻辑的自动化识别；最后，通过实验验证了该方法在实际场景中的有效性，准确率与召回率分别达到92.7%和89.4%。本研究不仅为时序数据库错误检测提供了新思路，还对提升系统可靠性和用户体验具有重要价值。",
              "关键词": ["时序数据库", "查询处理", "逻辑错误检测", "图神经网络", "语义分析"],
              "第一章：引言": {
                "研究背景": "随着物联网、智能设备的普及，时序数据库在处理海量时间序列数据方面发挥着关键作用。然而，由于时序查询逻辑复杂性高，开发人员常因对时间语义理解不足导致查询逻辑错误，进而影响系统稳定性、数据准确性与用户体验。",
                "文献综述": "国内外学者在数据库错误检测领域开展了广泛研究，主要包括SQL语法错误检测、查询优化错误检测等。然而，针对时序数据库特有的时间语义、滑动窗口操作、时间范围约束等查询逻辑的错误检测研究仍较为薄弱。",
                "研究问题与目标": "当前时序数据库查询逻辑错误检测存在检测范围窄、准确率低、依赖人工经验等问题。本研究旨在构建一个能够自动识别时序查询中逻辑错误的检测模型，提升时序数据库的运行可靠性。",
                "研究方法": "本论文采用基于语义分析与特征学习的混合方法，结合图神经网络（GNN）与时间序列特征提取技术，设计并实现了一种高效的错误检测模型。",
                "论文结构": "论文共分为五章，第一章为引言，第二章为文献综述，第三章阐述研究方法与技术路线，第四章进行实证分析，第五章总结研究成果并展望未来研究方向。"
              }
            }
            """
        article_framework = framework_agent.deserialization_article(json_res)
    except Exception as e:
        print(e)

    updated_paper = Chapter(topic=article_framework.topic, title=article_framework.topic)
    updated_paper.subsections = writing_content(article_framework, "")
    print(updated_paper.display())

    # 逐段评分
    updated_score = Chapter(topic=article_framework.topic, title=article_framework.topic)
    updated_score.subsections = article_score(updated_paper, "")
    print(updated_score.display())
