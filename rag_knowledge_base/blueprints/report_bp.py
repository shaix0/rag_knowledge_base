from flask import Blueprint, jsonify, render_template, request
from sqlalchemy import func
from rag_knowledge_base.models import db, Question

report_bp = Blueprint('report_bp', __name__, url_prefix='/report')

@report_bp.route('/top-errors')
def top_errors():
    """返回所有題目中錯誤率最高的題目列表"""
    limit = int(request.args.get('limit', 10))

    questions = Question.query\
        .filter(Question.error_count > 0)\
        .order_by(Question.error_count.desc())\
        .limit(limit)\
        .all()

    result = [{
        "id": q.id,
        "question_text": q.question_text,
        "error_count": q.error_count,
        "source": q.source or "未分類"
    } for q in questions]

    return render_template(
        'report.html',
        title='錯題統計',
        report_data=result,
        )
    return jsonify(result)

@report_bp.route('/top-errors-by-source')
def top_errors_by_source():
    """返回每個來源中錯誤率最高的題目列表"""

    subquery = db.session.query(
        Question.source,
        func.max(Question.error_count).label("max_error")
    ).group_by(Question.source).subquery()

    results = db.session.query(Question)\
        .join(
            subquery,
            (Question.source == subquery.c.source) &
            (Question.error_count == subquery.c.max_error)
        )\
        .all()

    data = [{
        "id": q.id,
        "question_text": q.question_text,
        "error_count": q.error_count,
        "source": q.source or "未分類"
    } for q in results]

    return jsonify(data)