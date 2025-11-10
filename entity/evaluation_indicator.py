# 文章内容实体对象
import json
from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class EvaluationDimension(str, Enum):
    vocab_professionalism = "语句用词专业度",
    logical_structure = "段落逻辑结构",
    use_of_transitions = "过渡语句的使用",
    fluency = "语言流畅度",
    redundancy_level = "内容重复度",


# 文章内容评分
class Indicator(BaseModel):
    dimension: EvaluationDimension
    score: float = Field(..., description="评分，满分 10 分")
    remarks: str = Field(...)

    # '{"评估维度":"语句用词专业度","评分":9.0,"详细评价":"整体用词专业且符合学术论文规范，术语使用准确，如‘时序数据库’‘查询表达式错误’‘动态特征提取模型’‘时间戳一致性’等均体现较强的领域专业性。对‘支持向量机（SVM）’‘随机森林’等机器学习算法的提及也符合技术表述习惯。但‘复杂性引发的误报’存在概念模糊——‘误报’通常用于检测系统输出，而此处描述的是逻辑错误的产生原因，建议改为‘导致逻辑判断偏差’或‘增加错误可能性’以提升术语严谨性。"}'
    # '{"评语"：[{"评估维度":"语句用词专业度"...}]}'
    # '[{"评估维度":"语句用词专业度"...}]'
    def parse_indicator_json(parsed_data: dict) -> List['Indicator']:
        try:
            # 先尝试解析为字典
            if isinstance(parsed_data, list):
                # 情况 3: 单个字典的列表
                return [
                    Indicator(**{field_mapping[key]: value for key, value in val.items()})
                    for val in parsed_data
                ]
            elif isinstance(parsed_data, dict):
                if "评语" in parsed_data:
                    # 情况 2: 包含对象列表的 JSON 对象
                    return [
                        Indicator(**{field_mapping[key]: value for key, value in val.items()})
                        for val in parsed_data['评语']
                    ]
                else:
                    # 情况 1: 单个 JSON 对象
                    return [Indicator(**{field_mapping[key]: value for key, value in parsed_data.items()})]
        except json.JSONDecodeError:
            print("无效的 JSON 字符串")
            return []


field_mapping = {
    '评估维度': 'dimension',
    '评分': 'score',
    '详细评价': 'remarks'
}

# 示例使用
if __name__ == "__main__":
    json_str1 = '{"评估维度":"语句用词专业度", "评分":9.0, "详细评价":"整体用词专业且符合学术论文规范，术语使用准确..."}'  # 情况 1
    json_str2 = '{"评语":[{"评估维度":"语句用词专业度", "评分":9.0, "详细评价":"整体用词专业且符合学术论文规范，术语使用准确..."}]}'  # 情况 2
    json_str3 = '[{"评估维度":"语句用词专业度", "评分":9.0, "详细评价":"整体用词专业且符合学术论文规范，术语使用准确..."}]'  # 情况 3

    # 测试：
    indicators1 = Indicator.parse_indicator_json(json_str1)
    indicators2 = Indicator.parse_indicator_json(json_str2)
    indicators3 = Indicator.parse_indicator_json(json_str3)

    # 输出结果
    print(indicators1)
    print(indicators2)
    print(indicators3)
