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
        title='Home Page',
        year=datetime.now().year,
        results=None
    )

@app.route('/search')
def search():
    """處理搜尋請求的路由。"""
    query = request.args.get('q', '').lower()

    if query:
        search_results = []
        for item in KNOWLEDGE_BASE:
            # 直接使用 JSON 檔案中已經整理好的「題目」和「選項」
            # 這裡只篩選出題目包含關鍵字的項目
            if "題目" in item and query in item["題目"].lower():
                # 將題目和選項添加到結果列表
                search_results.append({
                    "question_text": item.get("題目", ""),
                    "options": item.get("選項", [])
                })
    else:
        search_results = []

    return render_template(
        'index.html',
        title='搜尋結果',
        year=datetime.now().year,
        results=search_results,
        search_query=query
    )

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact',
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
