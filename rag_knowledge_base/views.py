"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template, request, jsonify
from rag_knowledge_base import app
import json
import os
import re
import numpy as np
from mistralai import Mistral
import random

#json_path = os.path.join(os.path.dirname(__file__), 'information', '醫學資訊管理師', 'questions_with_embeddings.json')
json_path = os.path.join(os.path.dirname(__file__), 'information', '醫學資訊管理師', 'all_questions_with_tags.json')
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

'''
@app.route('/search')
def search():
    tags=get_tag()
    search_type = request.args.get('searchtype')
    book_source = request.args.get('booksource', '')
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
            
            book_match = (not book_source or item.get("來源書籍", "") == book_source)

            if match and book_match:
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
        source_path=os.path.join(os.path.dirname(__file__), 'information', '醫學資訊管理師'),
        all_tags=tags
    )'''

@app.route('/search')
def search():
    tags = get_tag()
    search_type = request.args.get('searchtype')
    query = request.args.get(search_type, '').lower()

    if query:
        result_word = "查詢結果"
        search_results_raw = []
        
        # 判斷是否使用向量搜尋
        if search_type == 'q':
            # 呼叫向量搜尋函式
            vector_results = vector_search(query, similarity_threshold=0.7)
            
            # 將向量搜尋結果轉換為符合模板的格式
            for result in vector_results:
                item = result['item']
                search_results_raw.append({
                    "question_text": item.get("題目", ""),
                    "options": item.get("選項", []),
                    "book_source": item.get("來源書籍", ""),
                    "page_number": item.get("頁次", ""),
                    "source_filename": item.get("來源檔案", ""),
                    "answer": item.get("答案", ""),
                    "score": result['score'] # 可以顯示分數以便除錯
                })
        else:
            # 保留舊有的關鍵字或標籤搜尋邏輯
            # 這裡您可以根據需要自行調整
            pass

        if not search_results_raw:
            result_word = "查無結果"
            
    else:
        search_results_raw = []
        result_word = ""

    return render_template(
        'index.html',
        title='搜尋結果',
        year=datetime.now().year,
        results=search_results_raw,
        search_query=query,
        result_word=result_word,
        source_path=os.path.join(os.path.dirname(__file__), 'information', '醫學資訊管理師'),
        all_tags=tags
    )
# 餘弦相似度=點積=歐幾里得距離
def cosine(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    # 避免分母為零
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)
# 進行向量搜尋
def vector_search(query_text, similarity_threshold=0.5):
    # 檢查 API 金鑰是否已設定
    api_key = os.environ.get("MISTRAL_API_KEY")

    client = Mistral(api_key=api_key)
    model = "mistral-embed"

    try:
        # 生成查詢向量
        query_embedding_response = client.embeddings.create(
            model=model,
            inputs=[query_text],
        )
        query_vector = query_embedding_response.data[0].embedding
    except Exception as e:
        print(f"生成查詢向量時發生錯誤: {e}")
        return []

    # 載入所有題目的資料和嵌入向量
    questions_data = KNOWLEDGE_BASE
    if not questions_data:
        return []

    search_results = []

    # 遍歷所有題目，計算相似度並進行過濾
    for item in questions_data:
        if "embedding" in item:
            item_vector = item["embedding"]
            similarity_score = cosine(query_vector, item_vector) 
            
            # 只保留相似度大於或等於門檻值的項目
            if similarity_score >= similarity_threshold:
                 search_results.append({
                    "item": item,
                    "score": similarity_score
                })

    # 依據相似度分數進行排序（分數越高越好）
    search_results.sort(key=lambda x: x["score"], reverse=True)

    # 返回最相關的 top_n 個結果
    return search_results

@app.route('/edit')
def edit():
    """Renders the about page."""
    all_questions = []

    for item in KNOWLEDGE_BASE:
        all_questions.append({
                    "question_text": item.get("題目", ""),
                    "options": item.get("選項", []),
                    "book_source": item.get("來源書籍", ""),
                    "page_number": item.get("頁次", ""),
                    "source_filename": item.get("來源檔案", ""),
                    "answer": item.get("答案", "")
                })

    return render_template(
        'edit.html',
        title='題庫管理',
        year=datetime.now().year,
        results=all_questions
    )
   
@app.route('/upload_file', methods=['POST'])
def upload_file():
    file_path = os.path.join(os.path.dirname(__file__), 'information', '醫學資訊管理師')
    files = request.files.getlist("files")
    uploaded_paths = []

    # 1. 存檔
    for file in files:
        save_path = os.path.join(file_path, file.filename)
        file.save(save_path)
        uploaded_paths.append(save_path)

    # 2. 呼叫 pdftojson.py 腳本
    import rag_knowledge_base.pdftojson

    return jsonify({
        'message': f'成功上傳 {len(uploaded_paths)} 個檔案並呼叫處理函式。',
        'processed_files_count': len(uploaded_paths)
    }), 200

@app.route('/update_question', methods=['POST'])
def update_question():
    try:
        # 1. 接收前端 JSON 數據 (包含修改後的題目和索引)
        payload = request.get_json()
        
        if not payload or 'original_index' not in payload or 'question_data' not in payload:
            return jsonify({"error": "無效的 JSON 資料或缺少必要欄位"}), 400

        original_index = payload['original_index']
        modified_question_data = payload['question_data']
        
        # 2. 讀取當前的總體 JSON 檔案
        with open(json_path, 'r', encoding='utf-8') as f:
            all_questions = json.load(f)
            
        if not isinstance(all_questions, list):
            return jsonify({"error": "總體 JSON 檔案格式錯誤，不是列表"}), 500

        # 3. 執行數據覆寫
        if 0 <= original_index < len(all_questions):
            # 替換掉總體列表中的舊題目
            all_questions[original_index] = modified_question_data
            
            # 4. 寫回 JSON 檔案
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(all_questions, f, ensure_ascii=False, indent=4)
                
            return jsonify({
                "message": f"題目更新成功！ (第 {original_index + 1} 題)",
                "question_id": original_index
            }), 200
        else:
            return jsonify({"error": f"索引 {original_index} 超出範圍"}), 400

    except FileNotFoundError:
        return jsonify({"error": "找不到題目 JSON 檔案"}), 500
    except json.JSONDecodeError:
        return jsonify({"error": "JSON 檔案格式錯誤或接收數據異常"}), 400
    except Exception as e:
        print(f"更新題目時發生意外錯誤: {e}")
        return jsonify({"error": f"伺服器錯誤: {str(e)}"}), 500

@app.route('/toggle_favorite', methods=['POST'])
def toggle_favorite():
    # 1. 接收前端傳來的題目 JSON 數據
    question_data = request.get_json()
    
    client_question_data = request.get_json()
    
    # 2. 定義匹配鍵 (使用題幹和答案作為唯一識別)
    match_key = {
        'question_text': client_question_data.get('question_text'),
        'answer': client_question_data.get('answer'),
    }

    if not all(match_key.values()):
        return jsonify({"error": "缺少匹配題目所需的關鍵欄位 (question_text 或 answer)"}), 400

    all_questions = KNOWLEDGE_BASE
        
    try:
        found = False
        new_favorite_state = False

        # 4. 尋找並切換匹配題目的收藏狀態
        for question in all_questions:
            # 檢查題幹和答案是否匹配
            if (question.get('question_text') == match_key['question_text'] and
                    question.get('answer') == match_key['answer']):
                
                # 取得或初始化收藏狀態，並進行切換
                current_state = question.get('is_favorite', False)
                new_favorite_state = not current_state
                question['is_favorite'] = new_favorite_state
                found = True
                break # 找到後立即退出迴圈

        if not found:
            return jsonify({"error": "找不到匹配的題目進行更新"}), 404
        
        # 5. 將更新後的數據寫回 JSON 檔案
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_questions, f, ensure_ascii=False, indent=4)
        
        action = "收藏" if new_favorite_state else "取消收藏"
        return jsonify({
            "success": True, 
            "message": f"{action}成功",
            "is_favorite": new_favorite_state
        }), 200

    except FileNotFoundError:
        return jsonify({"error": f"JSON 檔案未找到: {json_path}"}), 500
    except json.JSONDecodeError:
        return jsonify({"error": "JSON 檔案格式錯誤，無法解析"}), 500
    except Exception as e:
        print(f"處理錯誤: {e}")
        return jsonify({"error": f"伺服器內部錯誤: {e}"}), 500
    
    return jsonify({"message": "收藏狀態已更新", "question": question_data})

app.config['SECRET_KEY'] = 'super_secret_key_for_quiz_app' 
TEMP_QUIZ_RESULTS = {} # 用於臨時存儲測驗結果
@app.route('/quiz')
def quiz():
    """ 
    渲染測驗頁面 (quiz.html)。
    根據 mode 參數載入不同的題目集。
    """
    mode = request.args.get('mode', 'selection') 

    if mode == 'selection':
        return render_template('quiz.html', 
                               title='選擇測驗模式', 
                               quiz_data=None, 
                               mode='selection')

    all_questions = KNOWLEDGE_BASE
    quiz_data = []
    title = '隨機模擬測驗'
    error_message = None

    if mode == 'practice':
        title = '錯題重練模式'
        # 篩選邏輯：選擇 error_count > 0 的題目
        quiz_data = [q for q in all_questions if q.get('error_count', 0) > 0]
        
        if not quiz_data:
            error_message = '目前沒有答錯的題目，請選擇隨機出題。'

    elif mode == 'weakness':
        title = '弱點加強模式'
        # 模擬弱點加強邏輯：選擇錯誤次數最多的題目
        sorted_questions = sorted(all_questions, key=lambda x: x.get('error_count', 0), reverse=True)
        # 取錯誤次數最多的前 5 題 (或所有有錯的題目)
        quiz_data = [q for q in sorted_questions if q.get('error_count', 0) > 0][:5]
        
        if not quiz_data:
             error_message = '目前沒有答錯的題目，請選擇隨機出題。'
    
    elif mode == 'normal': 
        # 隨機模擬測驗邏輯
        quiz_data = all_questions

    
    if error_message:
        return render_template('quiz.html', 
                               title='無法開始測驗', 
                               quiz_data=None, 
                               mode='selection',
                               error_message=error_message)

    # 將題目順序打亂
    random.shuffle(quiz_data)
    
    return render_template('quiz.html', 
                           title=title, 
                           quiz_data=quiz_data, 
                           mode=mode)

@app.route('/submit_quiz', methods=['POST'])
def submit_quiz():
    """
    接收使用者答案，計算成績，並返回詳細的評分結果，供前端即時顯示。
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    user_answers = data.get('answers', {})
    quiz_data = data.get('quiz_data', []) # 從前端獲取題目數據以進行比對
    
    # 參數設定
    SCORE_PER_QUESTION = 2
    correct_count = 0
    total_score = 0
    total_questions = len(quiz_data)
    
    quiz_results_for_report = []
    book_stats = {} 

    for index, question in enumerate(quiz_data, 1):
        q_key = f'question-{index}'
        user_answer = user_answers.get(q_key, None)
        # ⚠️ 注意：這裡的 '答案' 必須與 KNOWLEDGE_BASE 中的鍵名一致
        correct_answer = question.get('答案') 
        book_source = question.get('來源書籍', '未知章節')
        
        is_correct = (user_answer == correct_answer)

        if is_correct:
            correct_count += 1
            total_score += SCORE_PER_QUESTION
        
        # 彙整給報告頁面和前端即時顯示的數據
        quiz_results_for_report.append({
            "question_index": index,          
            "question_text": question.get('題目', 'N/A'),
            "book_source": book_source,
            "is_correct": is_correct,
            "user_answer": user_answer,
            "correct_answer": correct_answer, # 必須傳回給前端
        })
        
        # 統計章節數據 (略)
        if book_source not in book_stats:
            book_stats[book_source] = {"total": 0, "correct": 0}
        book_stats[book_source]["total"] += 1
        if is_correct:
            book_stats[book_source]["correct"] += 1
            
    # 總結結果
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": total_questions,
        "correct_count": correct_count,
        "total_score": total_score,
        "score_per_question": SCORE_PER_QUESTION,
        "overall_accuracy": (correct_count / total_questions * 100) if total_questions > 0 else 0,
        "detailed_results": quiz_results_for_report,
        "book_stats": book_stats
    }
    
    # 將結果儲存到臨時變數，以便報告頁面存取
    TEMP_QUIZ_RESULTS['latest_report'] = report_data

    # 返回完整的結果，供前端即時顯示
    return jsonify({
        "success": True,
        "total_score": total_score,
        "correct_count": correct_count,
        "total_questions": total_questions,
        "score_per_question": SCORE_PER_QUESTION,
        "detailed_results": quiz_results_for_report, # 關鍵：將詳細結果傳回
        "message": "測驗提交成功，結果已載入。"
    })

@app.route('/report_page')
def report_page():
    """渲染測驗報告頁面，載入最新計算的成績數據。"""
    report_data = TEMP_QUIZ_RESULTS.get('latest_report')
    
    if not report_data:
        # 如果沒有數據，導回測驗頁面
        return redirect(url_for('quiz'))
        
    # 將詳細的結果列表傳遞給報告模板
    return render_template('quiz_report.html', quiz_results=report_data['detailed_results'])

@app.route('/history_analysis')
def history_analysis():

    return render_template(
        'history_analysis.html',
        title='錯題分析',
        year=datetime.now().year,
        results=None,
        knowledge_base=KNOWLEDGE_BASE
    )

@app.route('/api/knowledge_base', methods=['GET'])
def api_knowledge_base_get():
    """
    API: 回傳完整的知識庫數據 (供前端分析使用)。
    """
    # 直接從服務層獲取數據
    data = quiz_service.get_knowledge_base()
    return jsonify(data)

@app.route('/api/knowledge_graph', methods=['GET'])
def api_knowledge_graph_get():
    """
    回傳知識圖譜資料 (nodes, links) 給前端 D3.js 使用。
    """
    try:
        # 以 KNOWLEDGE_BASE 為資料來源
        nodes = []
        links = []
        node_ids = set()

        # 1. 題目節點
        for idx, q in enumerate(KNOWLEDGE_BASE):
            q_id = f"q{idx}"
            nodes.append({
                "id": q_id,
                "name": q.get("題目", ""),
                "type": "question",
                "group": "question",
                "size": 10 + (q.get("error_count", 0) * 2),
                "error_count": q.get("error_count", 0)
            })
            node_ids.add(q_id)

            # 2. 標籤節點與連結
            for tag in q.get("標籤", []):
                tag_id = f"tag_{tag}"
                if tag_id not in node_ids:
                    nodes.append({
                        "id": tag_id,
                        "name": tag,
                        "type": "tag",
                        "group": "tag",
                        "size": 16
                    })
                    node_ids.add(tag_id)
                links.append({
                    "source": q_id,
                    "target": tag_id,
                    "value": 1
                })

            # 3. 書籍節點與連結
            book = q.get("來源書籍", "未知")
            book_id = f"book_{book}"
            if book_id not in node_ids:
                nodes.append({
                    "id": book_id,
                    "name": book,
                    "type": "book",
                    "group": "book",
                    "size": 22
                })
                node_ids.add(book_id)
            links.append({
                "source": q_id,
                "target": book_id,
                "value": 1
            })

        return jsonify({"nodes": nodes, "links": links})
    except Exception as e:
        print(f"知識圖譜 API 發生錯誤: {e}")
        return jsonify({"error": "知識圖譜資料生成失敗", "message": str(e)}), 500
