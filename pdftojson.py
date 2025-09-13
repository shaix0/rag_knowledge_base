import fitz  # PyMuPDF
import json
import re
import os

def parse_questions_from_text(text, exam_date, source_filename):
    """
    解析原始文本以提取題目、答案、出處等資訊，並將元數據加入其中。
    """
    questions = []
    
    # 正規表達式模式來匹配題目
    pattern = re.compile(
        r'\((\w)\)\s*(\d+)\.\s*(.*?)\s*(\(\*{1,2}.*?\)| \(時事\)| \(時事.*?\)| \(.*?\s*時事\)| \(時事\))\s*((?:\(.\).*?)+?)(?=\s*\(\w\)\s*\d|\s*\Z)',
        re.S | re.M
    )
    
    # 處理不規則換行，將選項前的換行替換為空格
    processed_text = re.sub(r'\n(?=\s*\(\w\))', ' ', text)
    
    matches = re.finditer(pattern, processed_text)

    for match in matches:
        try:
            answer = match.group(1).strip()
            # question_number = match.group(2).strip()
            question_text_and_options = match.group(3).strip() + " " + match.group(5).strip()
            source_info_str = match.group(4).strip().lstrip('(').rstrip(')')

            # 根據標記處理出處和頁次
            book_source = "無"
            page_number = "無"

            if '**' in source_info_str:
                book_source = "常用醫護術語"
                page_match = re.search(r'\*\*(.*)', source_info_str)
                if page_match:
                    page_number = page_match.group(1).strip().replace(')', '').strip()
            elif '*' in source_info_str:
                book_source = "醫學資訊管理學"
                page_match = re.search(r'\*(.*)', source_info_str)
                if page_match:
                    page_number = page_match.group(1).strip().replace(')', '').strip()
            
            if '時事' in source_info_str:
                if book_source == "醫學資訊管理學":
                    book_source += "、時事"
                else:
                    book_source = "時事"

            questions.append({
                "題目": question_text_and_options,
                "考試時間": exam_date,
                "出自哪一本書": book_source,
                "頁次": page_number,
                "答案": answer,
                "來源檔案": source_filename
            })
        except Exception as e:
            print(f"解析題目時發生錯誤: {e}. 原始匹配文字: {match.group(0)}")
            continue

    return questions

# 確保已安裝 PyMuPDF
# !pip install PyMuPDF

# --- 新增的程式碼部分：自動讀取資料夾內所有 PDF 檔案 ---

# 請將這裡的路徑替換為你存放 PDF 檔案的實際資料夾
# 如果是在 Google Colab 中，你需要先掛載 Google Drive
# 例如：pdf_folder_path = '/content/drive/MyDrive/my_exam_papers'
pdf_folder_path = os.path.join(os.path.dirname(__file__), 'rag_knowledge_base', 'information', '醫學資訊管理師')

# 獲取資料夾中所有以 .pdf 結尾的檔案
try:
    pdf_files = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]
except FileNotFoundError:
    print(f"錯誤: 找不到指定的資料夾 '{pdf_folder_path}'")
    pdf_files = []

# --- 核心邏輯保持不變 ---

all_extracted_questions = []

# 遍歷每個 PDF 檔案
for pdf_path in pdf_files:
    filename = os.path.basename(pdf_path)
    # 從檔名提取考試時間
    exam_date = os.path.splitext(filename)[0]

    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        print(f"成功從 {filename} 提取文本。")

        # 使用解析函式處理文本
        questions_from_file = parse_questions_from_text(text, exam_date, filename)
        all_extracted_questions.extend(questions_from_file)

    except Exception as e:
        print(f"從 {filename} 提取文本或解析時發生錯誤: {e}")
        continue

# 將所有整理好的題目儲存為 JSON 檔案
output_folder = os.path.join(os.path.dirname(__file__),'rag_knowledge_base', 'information', '醫學資訊管理師')  # 指定輸出資料夾
output_filename = os.path.join(output_folder, "all_questions.json")  # 將檔案儲存到指定資料夾

# 確保輸出資料夾存在
os.makedirs(output_folder, exist_ok=True)

with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(all_extracted_questions, f, ensure_ascii=False, indent=4)

print(f"\n所有 {len(all_extracted_questions)} 個題目已成功整理並儲存至 {output_filename}")
