# 主 Agent：控制整个文章生成的流程
from chapter_writing_agent import writing_content
from coherence_cohesion_assessment import paragraph_score
from content_rewriting import paragraph_rewriting
from entity.article import Chapter
from framework_agent import build_framework, deserialization_article

if __name__ == '__main__':
    topic = "时序数据库查询处理逻辑错误检测技术"
    try:
        '1. 搭建全文框架'
        print("** Start building the framework. **")
        json_res = build_framework(topic)
        print(json_res)
        article_framework = deserialization_article(json_res, topic)
        print("-------------------------------------------------------------")

        '2. 依据框架填充内容'
        print("** Start fill in the content of the article. **")
        updated_paper = Chapter(topic=topic, title=topic)
        updated_paper.subsections = writing_content(article_framework, "")
        print(updated_paper.display())
        print("-------------------------------------------------------------")

        '3. 逐段对内容进行评分，并逐段依据评语修订内容(循环3轮)'
        print("** Start revise the article paragraph by paragraph. **")
        for i in range(3):
            updated_score = Chapter(topic=topic, title=topic)
            updated_score.subsections = paragraph_score(updated_paper, "")
            print(updated_score.display())

            rewrite_article = Chapter(topic=topic, title=topic)
            rewrite_article.subsections = paragraph_rewriting(updated_score, "")
            print(rewrite_article.display())

            updated_paper = rewrite_article

        print("-------------------------------------------------------------")

        '4. 每章节内容进行评分，并逐章节依据评语修订内容(循环1轮)'
        print("** Start revise the article chapter by chapter. **")

        print("-------------------------------------------------------------")

        '5. 压缩前后章节内容，基于上下文信息对内容进行评分，并对其修订(循环2轮)'
        print("-------------------------------------------------------------")

        '6. TODO 全局标准评判'
    except Exception as e:
        print(e)
