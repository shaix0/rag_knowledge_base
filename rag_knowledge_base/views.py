"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template, request
from rag_knowledge_base import app
import json
import os
import re

json_path = 'all_questions_with_keywords.json'# os.path.join(os.path.dirname(__file__), 'information', '醫學資訊管理師', "all_questions.json")
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


def get_tag():
    # 提取並去重所有標籤
    all_tags = set()
    for item in KNOWLEDGE_BASE:
        if '標籤' in item and isinstance(item['標籤'], list):
            for tag in item['標籤']:
                all_tags.add(tag)

    return sorted(list(all_tags))

@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    tags=get_tag()  # 呼叫 get_tag 函式以確保標籤被提取
    return render_template(
        'index.html',
        title='首頁',
        year=datetime.now().year,
        results=None,
        all_tags=tags
    )

@app.route('/search')
def search():
    search_type = request.args.get('searchtype')
    """處理搜尋請求的路由。"""
    query = request.args.get(search_type, '').lower()
    match = False

    if query:
        search_results = []
        result_word = "查詢結果"
        for item in KNOWLEDGE_BASE:
            # 題目或答案包含關鍵字就加入結果
            if ( search_type == 'q' and (
                ("題目" in item and query in item["題目"].lower()) or
                ("答案" in item and query in item["答案"].lower()) )
            ):
                match = True
                
            elif search_type == 'tag' :
                for tag in item["標籤"]:
                    if (query in tag.lower()) :
                        match = True

            if match :
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

def search_by_keyword(query):
    results = []
    for item in KNOWLEDGE_BASE:
        if (
            ("題目" in item and query in item["題目"].lower()) or
            ("答案" in item and query in item["答案"].lower())
        ):
            results.append({
                "question_text": item.get("題目", ""),
                "options": item.get("選項", []),
                "book_source": item.get("來源書籍", ""),
                "page_number": item.get("頁次", ""),
                "source_filename": item.get("來源檔案", ""),
                "answer": item.get("答案", ""),
                "keywords": item.get("關鍵字", [])
            })
    return results

def search_by_tag(search_tag):
    results = []
    for item in KNOWLEDGE_BASE:
        if search_tag in [tag.lower() for tag in item.get('關鍵字', [])]:
            results.append({
                "question_text": item.get("題目", ""),
                "options": item.get("選項", []),
                "book_source": item.get("來源書籍", ""),
                "page_number": item.get("頁次", ""),
                "source_filename": item.get("來源檔案", ""),
                "answer": item.get("答案", ""),
                "keywords": item.get("關鍵字", [])
            })
    return results

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )

@app.route('/edit')
def edit():
    """Renders the about page."""
    return render_template(
        'edit.html',
        title='Edit',
        year=datetime.now().year,
        message='Your application description page.'
    )
