from flask import Blueprint, jsonify, render_template, request, redirect, url_for
from rag_knowledge_base.models import db,Question

# 建立 Blueprint
questions_bp = Blueprint('questions_bp', __name__, url_prefix='/questions')

@questions_bp.route('/')
def get_questions():
    """Renders the question management page with all questions from DB."""    
    
    # 從資料庫抓取所有題目
    all_questions_db = Question.query.all()

    return render_template(
        'questions.html',
        title='題庫',
        questions=all_questions_db
    )

@questions_bp.route('/sources_list')
def get_sources():
    sources = db.session.query(Question.source)\
        .distinct()\
        .all()

    return jsonify([s[0] for s in sources if s[0]])

@questions_bp.route('/search')
def search_questions():
    """Search for questions based on a query."""
    keyword = request.args.get('q','').strip()
    source = request.args.get('source', '').strip()
    
    filters = []

    # 關鍵字
    if keyword:
        filters.append(
            Question.question_text.ilike(f"%{keyword}%")
        )

    # 題目來源
    if source:
        filters.append(
            Question.source == source
        )

    # 建立 query
    query = Question.query

    # 套用條件（AND）
    if filters:
        query = query.filter(*filters)

    # 最後才執行
    search_results = query.limit(100).all()


    # Perform the search (example implementation - replace with actual search logic)
    result_word = "查詢結果" if search_results else "查無結果"

    return render_template(
        'index.html',
        title='搜尋結果',
        questions=search_results,
        result_word=result_word
    )