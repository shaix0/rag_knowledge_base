"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template, request
from rag_knowledge_base import app
import json
import os

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
        results=None  # 初始頁面不顯示搜尋結果
    )

@app.route('/search')
def search():
    """處理搜尋請求的路由。"""
    # 從 URL 參數中獲取搜尋關鍵字 'q'
    query = request.args.get('q', '').lower()

    # 進行搜尋
    if query:
        # 篩選知識庫中與關鍵字相關的條目
        # 這裡的邏輯是，只要關鍵字在任一條目的「題目」或「關鍵字」列表中，就算匹配
        search_results = [
            item["題目"]
            for item in KNOWLEDGE_BASE
            if query in item["題目"].lower() #or any(query in kw.lower() for kw in item["關鍵字"])
        ]
    else:
        # 如果沒有輸入關鍵字，返回空列表
        search_results = []

    # 將搜尋結果傳遞給模板
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
