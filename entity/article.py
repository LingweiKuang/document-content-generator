# 文章内容实体对象
import json
from abc import ABC, abstractmethod
from typing import List, Union, Optional

from pydantic import BaseModel, Field


# 抽象基类: 文章
class Article(ABC, BaseModel):
    @abstractmethod
    def display(self) -> str:
        pass

    @abstractmethod
    def to_prompt(self) -> str:
        pass

    @abstractmethod
    def summary_key(self) -> str:
        pass


# 内容类
class Content(Article):
    summary: str = Field(..., description="Summary of the content.")
    text: str = Field(..., description="文章的具体内容文本")
    remark: Optional[str] = Field(description="文章评语", default="")
    isModifiable: bool = True

    def display(self) -> str:
        result = self.text
        result += "\n{}".format(self.remark) if len(self.remark) > 0 else ""
        return result

    def to_prompt(self) -> str:
        return "{}".format(self.text)

    def summary_key(self) -> str:
        return self.summary

    def get_text(self) -> str:
        return self.text

    def get_remark(self) -> str:
        return self.remark

    def is_modifiable(self) -> bool:
        return self.isModifiable


# 章节类
class Chapter(Article):
    title: str = Field(..., description="章节标题")
    topic: Optional[str] = Field(description="文章主题", default=None)
    subsections: List[Union['Chapter', Content]] = Field(default_factory=list, description="子章节或内容列表")

    def add_subsection(self, subsection: Union['Chapter', Content]) -> None:
        """添加子章节或内容"""
        self.subsections.append(subsection)

    def display(self) -> str:
        if self.topic is not None:
            result = f"文章主题: {self.topic}\n"
        else:
            result = f"章节标题: {self.title}\n"
        for subsection in self.subsections:
            result += subsection.display() + "\n"
        return result

    def deserialization(self, dict_content: dict):
        sections = []

        for key, value in dict_content.items():
            if isinstance(value, dict):
                chapter = Chapter(title=key)
                chapter.deserialization(value)
                sections.append(chapter)
            else:
                # 若为 关键词 则不变
                content = Content(summary=key, text=str(value), isModifiable=True if "关键词" != key else False)
                sections.append(content)

        self.subsections = sections

    def to_prompt(self) -> str:
        if self.topic is not None:
            return self.topic
        else:
            return self.title

    def summary_key(self) -> str:
        if self.topic is not None:
            return self.topic
        else:
            return self.title


# Pydantic模型支持递归类型
Chapter.model_rebuild()

# 示例使用
if __name__ == "__main__":
    json_string = '''
    {
      "Abstract": "Provide a brief overview of the research objectives, methods, main findings, and significance. The abstract should typically be between 300-500 words.",
      "Keywords": "List 3-5 keywords that reflect the main research content of the thesis.",
      "Chapter 1: Introduction": {
        "Research Background": "Briefly introduce the background, purpose, and significance of the research.",
        "Literature Review": "Summarize domestic and international research on the topic, highlighting the novelty of the current study.",
        "Research Problem and Objectives": "Clearly state the research questions, objectives, and scope of the thesis.",
        "Research Methods": "Briefly describe the main research methods used.",
        "Structure of the Thesis": "Outline the content of each chapter."
      },
      "Chapter 2: Literature Review": {
        "Historical background and development trends in the field of study": "Key domestic and international research outcomes, summarizing the viewpoints of different scholars.",
        "Gaps in current research and identifying the innovations of the present study": "Identify the gaps in current research and explain the innovations of the present study."
      },
      "Chapter 3: Research Methods and Technical Approach": {
        "Research Design": "Introduce the theoretical framework, hypotheses, and research models.",
        "Data Collection": "Describe the experimental design, surveys, data sources, etc.",
        "Data Analysis Methods": "Introduce the statistical methods, tools, or software used for data analysis.",
        "Technical Approach": "Present the technical roadmap or flowchart of the research."
      },
      "Chapter 4: Empirical Analysis": {
        "Data Analysis": "Describe the analysis process in detail and present key results.",
        "Discussion of Results": "Compare the analysis results with the hypotheses or previous research and discuss the differences.",
        "Summary": "Summarize the main findings from the empirical analysis."
      },
      "Chapter 5: Conclusion and Future Directions": {
        "Main Conclusions": "Summarize the research findings and address the research questions.",
        "Contributions": "Discuss the contributions of the research to the field.",
        "Limitations and Future Research": "Point out the limitations of the study and suggest possible directions for future research."
      },
      "References": "List all the references cited in the thesis according to the standard citation format.",
      "Appendices": "Include any necessary additional materials such as data, survey questionnaires, code, etc."
    }
    '''
    data = json.loads(json_string)
    print(type(data))

    topic = "The title should be concise and accurately reflect the research content."
    paper = Chapter(title=topic, topic=topic)
    paper.deserialization(data)
    print(paper)
