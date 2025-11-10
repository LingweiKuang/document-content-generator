import os

from langchain_openai import ChatOpenAI

from config import config_init

''' 提取文章段落中心思想 '''

config_init.init()
base_url = os.getenv('BASE_URL', "https://api.siliconflow.cn/v1")
api_key = os.getenv('API_KEY')
model = "Qwen/Qwen2.5-7B-Instruct"

llm = ChatOpenAI(
    model=model,
    api_key=api_key,
    base_url=base_url,
)

system_prompt = """
你是中国中科院院士，擅长精炼中国硕士学术论文的段落内容，提取核心思想并进行简洁概括。请逐段分析 **{原文}**，用简洁、精炼的语言提炼出每段的核心要点，确保逻辑清晰、信息不失真。

示例：
** 文章主题 **：基于机器学习的图像分类算法优化\n摘要\n

** 原文 **：
```随着图像分类领域的快速发展，深度学习技术，尤其是卷积神经网络（CNN），推动了该领域的进步。现代图像分类算法在许多基准数据集（如ImageNet）上已经达到了或超过了人类水平，广泛应用于安全监控、自动驾驶、医学图像分析和内容推荐等领域。预训练模型（如ResNet、VGG、Transformer变种等）的迁移学习能力使得即使在小规模数据集上也能实现高性能，显著加速了实际应用的部署。```

** 压缩文 **：
`深度学习（CNN）推动图像分类达到超人类水平（如ImageNet），并广泛应用于自动驾驶、医学等领域。迁移学习加速部署，在小数据集上也能确保高性能。`
"""


def extract_content(prefix_prompt: str, content: str) -> str:
    human_prompt = "** 文章主题 **：{} \n** 原文 **:```{}``` \n** 压缩文 **:\n".format(prefix_prompt, content)
    messages = [("system", system_prompt), ("human", human_prompt)]
    ai_msg = llm.invoke(messages)
    return ai_msg.content


if __name__ == '__main__':
    prefix_prompt = "文章主题: 时序数据库查询处理逻辑错误检测技术\n摘要"
    content = """
    时序数据库在物联网、金融交易、智能监控等领域的广泛应用为数据分析提供了强有力的支持，但查询处理中的逻辑错误会导致数据不一致、业务中断等问题，亟需有效的检测手段予以应对。本研究针对时序数据库查询处理逻辑错误检测技术展开系统性研究，提出了一种基于语义分析与特征学习的错误检测模型。首先，我们深入分析了时序查询逻辑错误的主要类型及其特征，包括误排序、未连接、不匹配过滤条件、时间和值域错误等。基于这些分析，构建了一个包含时序语义约束的模型，用以捕捉和验证查询语句的逻辑正确性。

    其次，我们设计了一种基于图神经网络的错误检测算法。该算法通过构建查询图结构，利用图神经网络自动学习查询模式和错误特征，实现了对复杂查询逻辑的高精度识别。具体而言，该算法首先将每个查询语句分解为一系列节点和边，形成查询图，然后通过图神经网络进行特征表示和逻辑关系分析，从而有效识别出潜在的逻辑错误点。
    
    通过在实际应用场景中进行大规模实验验证，该方法展现了高度的准确性和可靠性。实验结果表明，该方法能够达到92.7%的准确率和89.4%的召回率。这些数据充分说明了该方法在实际应用中的有效性，不仅能够及时发现查询逻辑错误，还能够在保证业务连续性的同时提升数据处理的可靠性。
    
    综上所述，本研究不仅为时序数据库错误检测提供了新的思路，还为提升系统整体可靠性和用户体验做出了重要贡献。通过采用语义分析与特征学习相结合的方法，有效地提升了时序查询处理逻辑错误检测的精度和效率，具有重要的理论意义和实用价值。
    """
    key_idea = extract_content(prefix_prompt, content)
    print(key_idea)
