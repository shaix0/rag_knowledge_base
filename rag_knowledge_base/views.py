"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template, request
from rag_knowledge_base import app
import json
import os
import re

json_path = os.path.join(os.path.dirname(__file__), 'information', '醫學資訊管理師', "all_questions.json")
try:
    with open(json_path, 'r', encoding='utf-8-sig') as f:
        KNOWLEDGE_BASE = json.load(f)
    print("成功載入知識庫。")
except FileNotFoundError:
    print(f"錯誤: 找不到知識庫檔案於 {json_path}")
    KNOWLEDGE_BASE = []
except json.JSONDecodeError as e:
    print(f"解析 JSON 檔案時發生錯誤: {e}")
    KNOWLEDGE_BASE = []

@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='首頁',
        year=datetime.now().year,
        results=None
    )

@app.route('/search')
def search():
    """處理搜尋請求的路由。"""
    query = request.args.get('q', '').lower()

    if query:
        search_results = []
        result_word = "查詢結果"
        for item in KNOWLEDGE_BASE:
            # 題目或答案包含關鍵字就加入結果
            if (
                ("題目" in item and query in item["題目"].lower()) or
                ("答案" in item and query in item["答案"].lower())
            ):
                search_results.append({
                    "question_text": item.get("題目", ""),
                    "options": item.get("選項", []),
                    "book_source": item.get("來源書籍", ""),
                    "page_number": item.get("頁次", ""),
                    "source_filename": item.get("來源檔案", ""),
                    "answer": item.get("答案", "")
                })
        if search_results == []:
            result_word = "查無結果"
    else:
        search_results = []

    return render_template(
        'index.html',
        title='搜尋結果',
        year=datetime.now().year,
        results=search_results,
        search_query=query,
        result_word=result_word,
        source_path=os.path.join(os.path.dirname(__file__), 'information', '醫學資訊管理師')
    )

@app.route('/edit')
def edit():
    return render_template(
        'edit.html',
        title='管理題庫',
        year=datetime.now().year,
        message='Your contact page.'
    )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )
