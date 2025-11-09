# 文章内容实体对象
import json
from enum import Enum

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


field_mapping = {
    '评估维度': 'dimension',
    '评分': 'score',
    '详细评价': 'remarks'
}

# 示例使用
if __name__ == "__main__":
    json_string = '''
    '''
    json_string = json_string.replace("", "")
    print(json_string)

    data = json.loads(json_string)
    indicator = Indicator(**data)
    print(indicator)
    print(indicator.dict())
