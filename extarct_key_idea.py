import os

from langchain_openai import ChatOpenAI

from config import config_init

''' 提取文章段落中心思想 '''

config_init.init()
base_url = os.getenv('BASE_URL', "https://api.siliconflow.cn/v1")
api_key = os.getenv('API_KEY')

llm = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    api_key=api_key,
    base_url=base_url,
)

# *MAKE SURE* to follow the following guidelines:
# 1. `\n\n` is the paragraph delimiter
system_prompt_en = """
You are a Chinese Academy of Sciences academician, skilled in summarizing the core ideas of master’s thesis paragraphs and condensing the content into concise key information. Your goal is to extract the core idea of **each paragraph** from a given paper and present it in a concise format. Please use clear and concise language to ensure no critical information is lost from the original text.

Example:
Original: The current status of image classification has seen milestone advancements, largely driven by the rapid evolution of Deep Learning, especially Convolutional Neural Networks (CNNs). Modern image classification algorithms now achieve, and often surpass, human-level performance on many benchmark datasets (such as ImageNet), leading to widespread applications in security monitoring, autonomous driving, medical image analysis, and content recommendation. The transfer learning capability of pre-trained models (like ResNet, VGG, Transformer variants, etc.) enables high performance even on small-scale datasets with limited resources, significantly accelerating practical deployment.

Content Compression:
Deep Learning (CNNs) drives image classification to superhuman performance (ImageNet), leading to broad applications (e.g., autonomous driving, medical). Transfer learning accelerates deployment, ensuring high performance even on small datasets.
"""

system_prompt = """
你是一位中国中科院院士，擅长对硕士学术论文进行段落中心思想总结，并能够流畅地将段落内容压缩为简洁的核心信息。你的目标是根据给定的论文段落提炼出**每个段落**的核心思想，并以简洁的格式呈现。请注意使用简洁、精炼的语言，确保不失原文的关键信息。

示例：
原文：随着图像分类领域的快速发展，深度学习技术，尤其是卷积神经网络（CNN），推动了该领域的进步。现代图像分类算法在许多基准数据集（如ImageNet）上已经达到了或超过了人类水平，广泛应用于安全监控、自动驾驶、医学图像分析和内容推荐等领域。预训练模型（如ResNet、VGG、Transformer变种等）的迁移学习能力使得即使在小规模数据集上也能实现高性能，显著加速了实际应用的部署。

内容压缩文：
深度学习（CNN）推动图像分类达到超人类水平（如ImageNet），并广泛应用于自动驾驶、医学等领域。迁移学习加速部署，在小数据集上也能确保高性能。
"""

human_prompt = """
Topic: Logical Error Detection in Query Languages of Time Series Databases
Chapter 1: Introduction\n- **Research Problem and Objectives**: 物联网设备、传感器网络和智慧城市计划的迅猛发展导致了大量时间序列数据的产生，这需要先进的数据库管理系统（DBMS）来高效地处理时序数据。传统的关系型数据库管理系统（RDBMS）并不适合处理时间序列数据，因为它们缺乏高效的时间索引、优化的查询处理和实时分析等专门功能。因此，专门的时间序列数据库（TSDBs）应运而生，提供了优化的存储和查询执行机制，以满足时间序列数据分析的独特需求。
尽管有这些进展，时间序列数据库查询中的有效错误检测仍然是一个至关重要但未被充分解决的挑战。时间序列查询中的逻辑错误可能导致错误的结果、误导性的决策以及资源浪费，尤其是在异常检测、预测性维护和预测等应用中。这些错误往往由于对查询语义的误解、时间区间的错误指定和使用不支持的时间函数而引入。一个稳健的错误检测机制对于确保查询结果的可靠性和有效性至关重要。
本论文的主要目标是识别和分类时间序列查询中的常见逻辑错误，开发一个可以集成到时间序列数据库中的错误检测框架，并通过严格的测试和实际场景模拟评估其有效性。论文的范围将重点关注在时间序列数据库中遇到的主要查询操作，包括聚合、过滤、联接和基于窗口的操作，同时考虑事务完整性和一致性的影响。通过这项研究，期望能够提高时间序列数据分析系统的整体性能和可信度。
"""

messages = [
    (
        "system",
        system_prompt,
    ),
    ("human", human_prompt),
]

if __name__ == '__main__':
    ai_msg = llm.invoke(messages)
    print(ai_msg)
