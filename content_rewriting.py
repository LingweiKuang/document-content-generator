import os
from typing import Union, List

from langchain_openai import ChatOpenAI

import framework_agent
from chapter_writing_agent import writing_content
from coherence_cohesion_assessment import article_score
from config import config_init
from entity.article import Chapter, Article, Content

''' 内容重写模型：采取 FIM 类型，大语言模型能够在文本的任意位置（而不仅仅是末尾）智能生成缺失内容的技术。能够同时关注和理解前缀与后缀提供的完整上下文信息，从而做出更准确的预测。 '''

config_init.init()
base_url = os.getenv('BASE_URL', "https://api.siliconflow.cn/v1")
api_key = os.getenv('API_KEY')
# model = "Tongyi-Zhiwen/QwenLong-L1-32B"
model = "Qwen/Qwen2.5-7B-Instruct"

llm = ChatOpenAI(
    model=model,
    api_key=api_key,
    base_url=base_url,
)

# 3. **{文章主题}** 仅辅助作用，**切勿**将其作为正文输出
system_prompt = """
你是一位中国中科院院士，擅长根据论文评语对硕士学术论文进行修订。你的目标是依据 **{专家评语(JSON)}** 改进 **{原文}**，使之更加简洁、流畅、专业，并增强逻辑性和衔接度。请仔细阅读评语并修订，此外，请**谨记**以下规则：
1. **仅**输出修订后的内容文本，**切勿**对其重新评分，或者作出评判性描述。
2. 如有必要，可重塑整个段落。

示例：
```
** 文章主题 **：基于机器学习的图像分类算法优化\n摘要\n

** 原文 **：
``` 深度学习的技术，特别是卷积神经网络（CNN），对图像分类有了很大的影响，推动了该领域的进步。现在，现代算法的表现比以前好，甚至有些可以达到或超过人类的识别水平。由于这些技术已经在很多领域得到了应用，比如安全监控、自动驾驶和医学图像分析等，所以它们被认为是很有前景的。然而，在实际应用中，仍然会遇到很多问题，比如数据集太小、标注费用高，或者无法很好地适应不同的数据集等。为了克服这些问题，人们开始使用一些预训练模型，这些模型已经通过迁移学习得到了优化，像ResNet、VGG和Transformer的变种都能帮助我们在小数据集上获得不错的表现。通过这种方法，很多实际应用的部署速度得到了加快。 ```

** 专家评语(JSON) **：
``` [{"评估维度":"语句用词专业度","评分":5/10,"详细评价":"文章中的措辞较为简化，缺乏学术性和深度。比如，'影响'和'推动了该领域的进步'是过于抽象和通俗化的表达，未能深入探讨技术背后的关键原理和机制。此外，使用了'比以前好'这种不明确的表述，缺乏对具体技术进展的精确描述。"},{"评估维度":"段落逻辑结构","评分":6/10,"详细评价":"文章的逻辑结构较为松散，段落之间的转接较为突兀。例如，段落从'达到或超过人类的识别水平'跳跃到'广泛应用'，没有足够的过渡，且应用领域的列举过于宽泛。接着再从应用问题转到'预训练模型'的解决方案，缺乏对问题的具体分析和铺垫，使得段落的逻辑关系较为松散，给人一种信息跳跃的感觉。"},{"评估维度":"过渡语句的使用","评分":4/10,"详细评价":"过渡语句显得非常薄弱，句子间的衔接生硬。例如，'由于这些技术已经在很多领域得到了应用'直接跳到应用实例的列举，缺乏对前后内容的引导和衔接。对于从技术突破转到实际问题和解决方案的转换，也缺乏有效的过渡语句，使得内容的流动性差。"},{"评估维度":"语言流畅度","评分":6/10,"详细评价":"文章语言较为平淡，缺乏句式和节奏的变化，给人一种平铺直叙的感觉。例如，'现代算法的表现比以前好'这样的句式过于简单，缺乏必要的修饰或具体的技术细节，影响了文章的可读性和深度。同时，句子结构较为单一，缺乏变化，未能形成良好的节奏感。"},{"评估维度":"内容重复度","评分":7/10,"详细评价":"文章中存在一定的冗余内容，尤其是一些表述上重复出现的信息。例如，'推动了该领域的进步'与后续'技术已经在很多领域得到了应用'重复了相似的内容，没有进一步深化技术本身的讨论。虽然内容没有直接重复，但信息密度较低，影响了整体的表现力。"}] ```

** 输出 **：
`深度学习技术，尤其是卷积神经网络（CNN）的快速发展，彻底革新了图像分类领域。现代算法在ImageNet等特定大规模视觉基准上达到了或超过了人类识别水平。这些突破性进展使其被广泛应用于安全监控、自动驾驶和医学图像分析等关键领域。然而，在实际部署场景中，往往面临着目标数据集规模小、标注成本高或泛化性不足等挑战。针对这些局限性，预训练模型（如ResNet、VGG、Transformer变种）的迁移学习能力成为关键的解决方案，通过知识迁移，使得即使在小规模数据集上也能实现高性能，显著加速了实际应用的部署。`
```
"""


def rewrite_content(prefix_prompt: str, content: str, remark: str) -> str:
    human_prompt = "** 文章主题 **：{} \n** 原文 **:```{}``` \n** 专家评语(JSON) **:```{}``` \n** 输出 **：\n".format(
        prefix_prompt, content, remark)
    messages = [("system", system_prompt), ("human", human_prompt)]
    ai_msg = llm.invoke(messages)
    return ai_msg.content


def article_rewriting(article: Article, prefix_prompt: str) -> List[Union[Chapter, Article]]:
    prefix_prompt += article.to_prompt() + "\n"
    # 对子节点进行重写
    updated_sections = []
    for section in article.subsections:
        if isinstance(section, Chapter):
            # 递归评分章节
            chapter = Chapter(title=section.summary_key())
            chapter.subsections = article_rewriting(section, prefix_prompt)
            updated_sections.append(chapter)
        else:
            # 遍历 article 进行重写 => 按照框架节点进行重写, 主题-章节-段落中心思想(可能包含多段内容)
            if section.isModifiable:
                updated_content = rewrite_content(prefix_prompt, section.to_prompt(), section.get_remark())
            else:
                updated_content = section.get_text()
            updated_content = updated_content.replace("```", "").replace("`", "")
            content = Content(summary=section.summary_key(), text=updated_content,
                              isModifiable=section.is_modifiable(), remark="")
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

    # 逐段填充内容
    updated_paper = Chapter(topic=article_framework.topic, title=article_framework.topic)
    updated_paper.subsections = writing_content(article_framework, "")
    print(updated_paper.display())

    # 逐段评分
    updated_score = Chapter(topic=article_framework.topic, title=article_framework.topic)
    updated_score.subsections = article_score(updated_paper, "")
    print(updated_score.display())

    # 逐段修订
    rewrite_article = Chapter(topic=article_framework.topic, title=article_framework.topic)
    rewrite_article.subsections = article_rewriting(updated_score, "")
    print(rewrite_article.display())
