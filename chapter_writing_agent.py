import os
from typing import List, Union

from langchain_openai import ChatOpenAI

import framework_agent
from config import config_init
from entity.article import Chapter, Article, Content

''' 论文章节行文 '''

config_init.init()
base_url = os.getenv('BASE_URL', "https://api.siliconflow.cn/v1")
api_key = os.getenv('API_KEY')
model = "Qwen/Qwen2.5-7B-Instruct"
# model = "Tongyi-Zhiwen/QwenLong-L1-32B"

llm = ChatOpenAI(
    model=model,
    api_key=api_key,
    base_url=base_url,
)

# TODO prompt 优化：**{原文}**
system_prompt = """
**【角色与身份设定】**
你现为中国科学院的资深院士，具有深厚的学术积累，尤其擅长撰写高质量的硕士学位论文。

**【核心任务】**
你将根据用户提供的**【核心研究点/段落说明】**，生成**当前指定章节**的完整学术论文内容。

**【输出规范与标准】**
1.  **内容深度与逻辑：**确保内容详实、逻辑严密。
2.  **语言风格：**必须使用**高度专业化、简洁、精准**的学术语言，杜绝任何口语化或冗余的描述。
3.  **学术严谨性：** 严格遵循学术规范。在需要引用处（如引用文献、数据支撑）应使用**占位符或指定格式**（例如：[1]、(Author, Year)），体现结构完整性，但**不要求实际生成文献列表**。
4.  **严格限制输出范围：** **只**输出当前请求的段落内容，**绝不**输出任何前言、总结、或与当前段落主题无关的引导性/解释性文字。

示例：

**特定段落的写作方法(输入)**：
```
题目：基于机器学习的图像分类算法优化
第一章：引言
**研究背景**：图像分类概述，准确率方面的挑战，机器学习的作用。
```

**该段的详细内容(输出)**：
```
目前的图像分类已经取得了里程碑式的进步，这在很大程度上是由深度学习，特别是卷积神经网络（cnn）的快速发展所推动的。现代图像分类算法现在在许多基准数据集（如ImageNet）上达到并经常超过人类水平的性能，从而在安全监控、自动驾驶、医学图像分析和内容推荐中得到广泛应用。预训练模型（如ResNet、VGG、Transformer变体等）的迁移学习能力即使在资源有限的小规模数据集上也能实现高性能，显著加快了实际部署。
尽管取得了这些进步，但图像分类仍然面临着精度方面的重大挑战。主要的困难包括由于数据不平衡造成的类别偏见、对Few-Shot学习的需求、模型对对抗性攻击的脆弱性，以及在复杂场景（如剧烈的光照变化、严重的遮挡或极端的视角）中保持鲁棒性。此外，缺乏模型可解释性限制了在医疗诊断等关键领域的信任和部署。
机器学习，特别是通过强化学习、元学习和更先进的深度学习架构等技术，在应对这些挑战方面发挥着关键作用。通过设计更精细的网络结构，整合注意力机制，利用生成对抗网络（GANs）进行数据增强，并应用可解释人工智能（XAI）技术揭示决策过程，机器学习方法正在不断优化现有算法的准确性、鲁棒性和泛化能力，以更好地解决复杂的现实世界分类问题。
```
"""


# 将结构化文章数据填充具体内容
def writing_content(article: Article, prefix_prompt: str) -> List[Union[Chapter, Article]]:
    prefix_prompt += article.to_prompt() + "\n"
    updated_sections = []
    # 层序遍历，将文章子节点依次通过 AI 填充
    for section in article.subsections:
        if isinstance(section, Chapter):
            chapter = Chapter(title=section.summary_key())
            chapter.subsections = writing_content(section, prefix_prompt)
            updated_sections.append(chapter)
        else:
            # 遍历 article 进行内容填充
            content_prompt = "```{}{}```".format(prefix_prompt, section.to_prompt())
            messages = [("system", system_prompt), ("human", content_prompt)]
            content = Content(summary=section.summary_key(),
                              text=llm.invoke(messages).content if section.is_modifiable() else section.get_text(),
                              isModifiable=section.is_modifiable())
            updated_sections.append(content)

    return updated_sections


if __name__ == '__main__':
    topic = "时序数据库查询处理逻辑错误检测技术"

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
          },
          "第二章：文献综述": {
            "研究领域的历史背景与发展趋势": "随着时间序列数据在各行业的广泛应用，研究人员逐步意识到时序查询逻辑错误的严重性。早期研究聚焦在语法检查，后续转向语义分析与逻辑验证。近年来，基于深度学习的方法逐渐成为主流，但多集中于通用数据库查询检测，缺乏针对时序数据库的专用研究。",
            "当前研究的不足与本研究的创新点": "现有研究缺乏对时序查询特有的时间窗口、时间连续性、时间语义一致性等逻辑错误的系统性分析。本研究首次提出基于时序语义约束的错误检测模型，结合图神经网络实现复杂查询逻辑的自动化识别，有效提升了检测精度与泛化能力。"
          },
          "第三章：研究方法与技术路线": {
            "研究设计": "构建时序语义约束模型，包括时间范围、时间连续性、滑动窗口等关键语义要素。设计查询逻辑错误分类体系，明确不同错误类别的判定标准。基于此，提出基于图神经网络的错误检测模型，将查询语句转化为图结构进行特征提取。",
            "数据收集": "从开源时序数据库项目（如InfluxDB、TimescaleDB）中提取真实查询样本，结合人工标注构建训练数据集。数据集包含10,000条查询语句，涵盖五类时序逻辑错误类型，每类2000条。为评估模型性能，额外采集3,000条测试数据。",
            "数据分析方法": "采用图神经网络（GNN）进行查询语句的图结构建模，结合注意力机制提取关键特征，并使用SVM分类器进行错误检测。通过交叉验证评估模型性能，以准确率、召回率、F1值为评价指标。",
            "技术路线": "技术路线包括四个关键环节：时序查询逻辑错误分类、语义约束规则提取、图结构建模与特征提取、基于深度学习的错误检测，形成系统性研究流程。"
          },
          "第四章：实证分析": {
            "数据分析": "对提取的10,000条数据进行了清洗与预处理，采用特征工程方法提取语义特征。基于GNN模型，模型在训练集上的准确率为94.3%，在测试集上的准确率为92.7%，召回率为89.4%，F1值为90.8%。",
            "结果讨论": "与传统语法检查方法相比，本模型在时序逻辑错误检测上准确率提升了18.5%，召回率提升12.3%。与现有的通用错误检测模型相比，本模型在检测时间窗口错误上的准确率提高了23.7%。",
            "小结": "实证分析表明，基于语义分析与图神经网络的错误检测模型能有效识别各种时序查询逻辑错误，显著提升检测精度，证明了本研究方法的可行性与实用性。"
          },
          "第五章：结论与未来研究方向": {
            "主要结论": "本研究系统性分析了时序数据库查询逻辑错误的类型与特征，提出了基于语义分析与图神经网络的错误检测模型，成功解决了时序查询中时间连续性、时间范围、滑动窗口等逻辑错误的检测问题。",
            "研究贡献": "在理论层面，构建了时序查询逻辑错误的分类体系和时序语义约束模型，丰富了数据库错误检测的理论框架；在实践层面，提供了一种高效、自动化的错误检测方法，为数据库管理系统提供技术支持。",
            "研究局限与未来研究": "目前模型主要针对结构化查询，在非结构化时序数据处理上的表现有待提升；未来可探索集成强化学习，提高系统对未知错误类型的学习能力；还可以结合领域知识图谱，实现面向特定应用领域的错误检测。"
          },
          "参考文献": "1. Wang, L., et al. (2021). 'Time Series Query Error Detection Using Deep Learning.' IEEE Transactions on Knowledge and Data Engineering. 2. Zhang, Y., et al. (2020). 'A Survey on Database Error Detection Techniques.' ACM Computing Surveys. 3. Chen, X., et al. (2022). 'Graph Neural Networks for Query Optimization in Time-Series Databases.' VLDB Journal. 4. Liu, J., & Li, H. (2023). 'Semantic Analysis of Time-Series Queries for Error Detection.' Proceedings of the ACM SIGMOD Conference. 5. Johnson, M., et al. (2019). 'Advanced SQL Error Checking in High-Performance Databases.' VLDB Endowment.",
          "附录": "附录A：时序查询逻辑错误分类体系表；附录B：语义约束规则集；附录C：数据集样本示例；附录D：模型训练参数设置。"
        }
        """
        article_framework = framework_agent.deserialization_article(json_res, "")
    except Exception as e:
        print(e)

    updated_paper = Chapter(topic=article_framework.topic, title=article_framework.topic)
    updated_paper.subsections = writing_content(article_framework, "")
    print(updated_paper.display())
