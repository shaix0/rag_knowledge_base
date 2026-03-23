import random
from sqlalchemy.sql.expression import func
from rag_knowledge_base.models import Question

QUIZ_SIZE = 50

def get_questions_by_mode(mode):
    query = Question.query

    if mode == "practice":
        query = query.filter(Question.error_count > 0)

    elif mode == "weakness":
        query = query.filter(Question.error_count > 0)\
                     .order_by(Question.error_count.desc())

    elif mode == "normal":
        pass  # 全部題目

    else:
        return []

    # 隨機抽 50 題
    questions = query.order_by(func.random()).limit(QUIZ_SIZE).all()
    # 轉成 dict 列表，前端安全使用
    questions_serializable = [q.to_dict() for q in questions]

    return questions_serializable