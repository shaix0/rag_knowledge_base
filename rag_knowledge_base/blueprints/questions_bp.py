from flask import Blueprint, render_template, request, redirect, url_for
from rag_knowledge_base.models import Question

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

@questions_bp.route('/search')
def search_questions():
    """Search for questions based on a query."""
    keyword = request.args.get('q','').strip()
    
    # 如果有輸入關鍵字就搜尋
    if keyword:
        search_results = Question.query.filter(
            Question.question_text.ilike(f"%{keyword}%")
        ).all()
    else:
        search_results = []

    # Perform the search (example implementation - replace with actual search logic)
    result_word = "查詢結果" if search_results else "查無結果"

    return render_template(
        'index.html',
        title='搜尋結果',
        questions=search_results,
        result_word=result_word
    )