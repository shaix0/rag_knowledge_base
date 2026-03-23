from random import random
from turtle import title

from flask import Blueprint, jsonify, render_template, request, redirect, url_for
from rag_knowledge_base.models import Question
from rag_knowledge_base.services.quiz_service import get_questions_by_mode
from rag_knowledge_base.config.quiz_config import MODE_CONFIG

# 建立 Blueprint
quiz_bp = Blueprint('quiz_bp', __name__, url_prefix='/quiz')

@quiz_bp.route('/')
def quiz():
    """ 根據 mode 參數載入不同的題目集 """
    mode = request.args.get('mode', 'selection') 
    if mode != 'selection':
        config = MODE_CONFIG.get(mode, MODE_CONFIG[mode])  # 預設為 selection 模式的配置:

    if mode == 'selection':
        return render_template(
            'quiz.html', 
            title='選擇測驗模式', 
            quiz_data=None, 
            mode='selection',
            modes=MODE_CONFIG.keys(),
            mode_configs=MODE_CONFIG
        )

    quiz_data = get_questions_by_mode(mode)

    if not quiz_data:
        return render_template(
            'quiz.html',
            title='無法開始測驗',
            quiz_data=None,
            mode='selection',
            error_message='沒有符合條件的題目'
        )
    
    return render_template(
        'quiz.html', 
        title=config['title'], 
        quiz_data=quiz_data, 
        mode=mode,
        icon=config['icon'],
        time_limit=config['time']
    )

TEMP_QUIZ_RESULTS = {}
@quiz_bp.route('/submit', methods=['POST'])
def submit_quiz():
    """
    接收使用者答案，計算成績，並返回詳細的評分結果，供前端即時顯示。
    注意：前端只傳題目 id，不再傳整個題目物件。
    """
    if not request.is_json:
        return jsonify({"success": False, "message": "Content-Type must be application/json"}), 400

    data = request.get_json()
    user_answers = data.get('answers', {})
    quiz_ids = data.get('quiz_data', [])  # 前端只傳題目 ID
    time_spent = data.get('time_spent', 0)

    if not quiz_ids:
        return jsonify({"success": False, "message": "沒有題目 ID"}), 400

    # 從資料庫查詢題目
    questions = Question.query.filter(Question.id.in_(quiz_ids)).all()

    SCORE_PER_QUESTION = 2
    correct_count = 0
    total_score = 0
    total_questions = len(questions)
    detailed_results = []
    book_stats = {}

    # 題目比對
    for index, q in enumerate(questions, 1):
        q_key = f'question-{index}'
        user_answer = user_answers.get(q_key)
        correct_answer = q.answers[0] if q.answers else None  # 假設第一個是標準答案
        book_source = q.source or "未知章節"

        is_correct = (user_answer == correct_answer)
        if is_correct:
            correct_count += 1
            total_score += SCORE_PER_QUESTION

        detailed_results.append({
            "question_index": index,
            "question_text": q.question_text,
            "book_source": book_source,
            "is_correct": is_correct,
            "user_answer": user_answer,
            "correct_answer": correct_answer
        })

        # 章節統計
        if book_source not in book_stats:
            book_stats[book_source] = {"total": 0, "correct": 0}
        book_stats[book_source]["total"] += 1
        if is_correct:
            book_stats[book_source]["correct"] += 1

    # 儲存報告到臨時變數
    TEMP_QUIZ_RESULTS['latest_report'] = {
        "total_questions": total_questions,
        "correct_count": correct_count,
        "total_score": total_score,
        "score_per_question": SCORE_PER_QUESTION,
        "overall_accuracy": (correct_count / total_questions * 100) if total_questions > 0 else 0,
        "detailed_results": detailed_results,
        "book_stats": book_stats
    }

    # 返回給前端
    return jsonify({
        "success": True,
        "total_score": total_score,
        "correct_count": correct_count,
        "total_questions": total_questions,
        "score_per_question": SCORE_PER_QUESTION,
        "detailed_results": detailed_results,
        "message": "測驗提交成功，結果已載入。"
    })