# -*- coding: utf-8 -*-
import os
import json
import random
import numpy as np
from datetime import datetime
from mistralai import Mistral # 引入 AI 模型
import hashlib

class QuizManager:
    """
    測驗與知識庫管理服務層。
    負責知識庫的載入、測驗生成、評分、錯誤計數更新、向量搜尋等核心業務邏輯。
    """
    
    # 設置每題分數為類別常數
    SCORE_PER_QUESTION = 2 

    def __init__(self, json_path=None, embedding_key="embedding"):
        """初始化服務，載入知識庫"""
        if json_path is None:
            json_path = os.path.join(
                os.path.dirname(__file__),
                'information', '醫學資訊管理師', 'all_questions_with_tags.json'
            )
        self.json_path = json_path
        self.embedding_key = embedding_key
        self.knowledge_base = []
        self.load_knowledge_base()

    def load_knowledge_base(self):
        """載入知識庫並初始化穩定 ID 和 error_count"""
        try:
            with open(self.json_path, 'r', encoding='utf-8-sig') as f:
                self.knowledge_base = json.load(f)
            
            # 為每道題目新增一個穩定的 ID 和 error_count
            for idx, item in enumerate(self.knowledge_base):
                # 使用問題題目生成穩定的 ID，避免依賴索引，如果題目重複則使用 index
                # 這裡為了演示和兼容性，我們仍使用索引作為 ID，但應警告其不穩定性
                item['id'] = f"q_{idx}" 
                item['error_count'] = item.get('error_count', 0)
                item['is_favorite'] = item.get('is_favorite', False)
                item['user_notes'] = item.get('user_notes', '')

            print(f"✅ 成功載入知識庫，共 {len(self.knowledge_base)} 題。並已初始化 ID/error_count。")
        except FileNotFoundError:
            print(f"❌ 找不到知識庫檔案於 {self.json_path}")
            self.knowledge_base = []
        except json.JSONDecodeError as e:
            print(f"❌ 解析 JSON 檔案時發生錯誤: {e}")
            self.knowledge_base = []

    def save_knowledge_base(self):
        """將知識庫的當前狀態（包含 error_count 等）存回檔案"""
        try:
            # 移除非持久化數據（如暫時 ID 或計算出的數據，但這裡只有 error_count 和 is_favorite，我們都存）
            with open(self.json_path, 'w', encoding='utf-8') as f:
                # 複製一份數據，避免將 'id' 寫入，因為 'id' 是載入時生成的，但考慮到後續應用，我們保持原樣。
                data_to_save = []
                for q in self.knowledge_base:
                    q_copy = q.copy()
                    q_copy.pop('id', None) # 儲存時移除載入時新增的 ID
                    data_to_save.append(q_copy)

                json.dump(data_to_save, f, ensure_ascii=False, indent=4)
            print(f"✅ 題庫已儲存至 {self.json_path}")
            return True
        except Exception as e:
            print(f"❌ 題庫儲存失敗: {e}")
            return False

    def get_tags(self):
        """獲取所有標籤列表"""
        all_tags = set()
        for item in self.knowledge_base:
            if '標籤' in item and isinstance(item['標籤'], list):
                for tag in item['標籤']:
                    all_tags.add(tag)
        return sorted(list(all_tags))

    def get_questions(self):
        """回傳完整的知識庫數據"""
        return self.knowledge_base

    def get_question_by_id(self, q_id):
        """根據 ID 獲取題目及其在知識庫中的索引"""
        for idx, q in enumerate(self.knowledge_base):
            if q.get('id') == q_id:
                return q, idx
        return None, -1

    def update_user_notes(self, q_id, user_notes):
        """根據 ID 更新用戶筆記"""
        q, idx = self.get_question_by_id(q_id)
        if idx != -1:
            self.knowledge_base[idx]['user_notes'] = user_notes
            return self.save_knowledge_base()
        return False

    def toggle_favorite(self, q_id):
        """根據 ID 切換收藏狀態"""
        q, idx = self.get_question_by_id(q_id)
        if idx != -1:
            current_state = q.get('is_favorite', False)
            self.knowledge_base[idx]['is_favorite'] = not current_state
            self.save_knowledge_base()
            return q['is_favorite']
        return None

    def vector_search(self, query_text, similarity_threshold=0.7):
        """
        AI 數據處理：執行向量相似度搜尋。
        所有 AI/Data 錯誤處理和模型調用都封裝在這裡。
        """
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
             print("❌ MISTRAL_API_KEY 未設定，無法執行向量搜尋。")
             return []

        client = Mistral(api_key=api_key)
        model = "mistral-embed"
        
        try:
            # 1. 生成查詢向量
            query_embedding_response = client.embeddings.create(
                model=model,
                inputs=[query_text],
            )
            query_vector = query_embedding_response.data[0].embedding
        except Exception as e:
            print(f"❌ 生成查詢向量時發生錯誤 (Mistral API Error): {e}")
            return []

        # 2. 進行相似度計算 (這是核心數據處理)
        search_results = []
        for item in self.knowledge_base:
            if self.embedding_key in item:
                item_vector = item[self.embedding_key]
                similarity_score = self.cosine(query_vector, item_vector)
                
                # 篩選和整理結果
                if similarity_score >= similarity_threshold:
                    # 避免將超長的 embedding 向量傳給前端
                    item_copy = item.copy()
                    item_copy.pop(self.embedding_key, None) 
                    search_results.append({
                        "item": item_copy,
                        "score": similarity_score
                    })
                    
        search_results.sort(key=lambda x: x["score"], reverse=True)
        return search_results

    @staticmethod
    def cosine(vec1, vec2):
        """計算餘弦相似度 (Cos-Sim)"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def generate_quiz(self, mode='normal', quiz_size=50):
        """
        根據模式生成測驗題目列表 (原 sample_quiz)。
        模式: 'normal' (隨機), 'practice' (錯題重練), 'weakness' (弱點加強/加權隨機)
        """
        all_questions = self.knowledge_base
        source_questions = []
        title = '隨機模擬測驗'
        error_message = None

        if mode == 'practice':
            title = '錯題重練模式'
            source_questions = [q for q in all_questions if q.get('error_count', 0) > 0]
            if not source_questions:
                error_message = '目前沒有答錯的題目，請選擇其他模式。'
        
        elif mode == 'weakness':
            title = '弱點加強模式'
            weighted_population = []
            weights = []
            
            # 僅根據錯誤次數進行加權，錯誤次數越多，被選中機率越高
            for q in all_questions:
                error_count = q.get('error_count', 0)
                if error_count > 0:
                    weighted_population.append(q)
                    weights.append(error_count * error_count) # 平方加權，拉大差異
            
            if not weighted_population:
                error_message = '目前沒有答錯的題目紀錄，請選擇隨機出題。'
                
            num_to_sample = min(quiz_size, len(weighted_population))
            final_quiz_data = []
            if num_to_sample > 0 and weighted_population:
                # 使用 set 確保選出的是不同 ID 的題目
                chosen_questions = set()
                while len(chosen_questions) < num_to_sample and len(final_quiz_data) < quiz_size * 5: # 設置防呆上限
                    chosen_question = random.choices(weighted_population, weights=weights, k=1)[0]
                    if chosen_question['id'] not in chosen_questions:
                        chosen_questions.add(chosen_question['id'])
                        final_quiz_data.append(chosen_question)
                        
            quiz_data = final_quiz_data
            
            # 從結果中移除答案
            quiz_data_no_answer = [q.copy() for q in quiz_data]
            for q in quiz_data_no_answer:
                q.pop('答案', None)
            
            return quiz_data_no_answer, title, error_message, quiz_data # 返回原始數據給後端儲存
            
        else: # 'normal' 模式 (隨機)
            title = '隨機模擬測驗'
            source_questions = all_questions

        num_to_sample = min(quiz_size, len(source_questions))
        quiz_data = random.sample(source_questions, num_to_sample) if num_to_sample > 0 else []
        
        # 從結果中移除答案
        quiz_data_no_answer = [q.copy() for q in quiz_data]
        for q in quiz_data_no_answer:
            q.pop('答案', None)

        return quiz_data_no_answer, title, error_message, quiz_data # 返回原始數據給後端儲存

    def process_quiz_submission(self, submitted_quiz_data, user_answers):
        """
        整合評分 (score_quiz) 和錯誤更新 (update_error_count) 邏輯。
        這是提交測驗時的核心業務邏輯。
        
        Args:
            submitted_quiz_data (list): 後端暫存的原始題目數據 (包含答案)。
            user_answers (dict): 使用者提交的答案 { 'question-1': 'A', ... }
            
        Returns:
            dict: 包含測驗報告數據。
        """
        correct_count = 0
        total_score = 0
        total_questions = len(submitted_quiz_data)
        quiz_results_for_report = []
        book_stats = {}
        
        # 1. 進行評分、更新錯誤計數
        for index, question in enumerate(submitted_quiz_data, 1):
            q_key = f'question-{index}' # 前端傳回的鍵
            user_answer = user_answers.get(q_key, None)
            correct_answer = question.get('答案')
            book_source = question.get('來源書籍', '未知章節')
            is_correct = (user_answer == correct_answer)
            
            # --- 核心數據處理：更新錯誤次數並持久化 ---
            if not is_correct:
                # 找到原始題目並更新 error_count
                # 注意：這裡使用問題標題作為鍵是不夠穩定的，但基於您原有的邏輯結構，我們暫時保留
                q_id = question.get('id')
                q_original, idx = self.get_question_by_id(q_id)
                if idx != -1:
                    self.knowledge_base[idx]['error_count'] = self.knowledge_base[idx].get('error_count', 0) + 1
            
            # 2. 計算分數和統計
            if is_correct:
                correct_count += 1
                total_score += self.SCORE_PER_QUESTION

            # 3. 準備報告詳細數據
            quiz_results_for_report.append({
                "question_id": index,
                "id": question.get('id'),
                "question_text": question.get('題目', 'N/A'),
                "book_source": book_source,
                "is_correct": is_correct,
                "user_answer": user_answer,
                "correct_answer": correct_answer,
            })
            
            # 4. 統計書籍答題情況
            if book_source not in book_stats:
                book_stats[book_source] = {"total": 0, "correct": 0}
            book_stats[book_source]["total"] += 1
            if is_correct:
                book_stats[book_source]["correct"] += 1

        # 5. 持久化錯誤計數 (保存到 JSON 檔案)
        self.save_knowledge_base() 
        
        # 6. 總結報告
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "total_questions": total_questions,
            "correct_count": correct_count,
            "total_score": total_score,
            "score_per_question": self.SCORE_PER_QUESTION,
            "overall_accuracy": (correct_count / total_questions * 100) if total_questions > 0 else 0,
            "detailed_results": quiz_results_for_report,
            "book_stats": book_stats
        }
        return report_data
