import fitz  # PyMuPDF
import json
import re
import os

def parse_questions_from_text(text, exam_date, source_filename):
    """
    解析原始文本以提取題目、答案、選項、出處等資訊，並將元數據加入其中。
    """
    questions = []

    # 處理不規則換行，將選項前的換行替換為空格，並統一移除題目文本中的換行符
    # 這裡使用更廣泛的模式來處理換行問題
    processed_text = re.sub(r'\n(?=\s*\(\w\))', ' ', text)
    processed_text = processed_text.replace('\n', ' ').replace('\r', '')
    
    # 重新設計更強健的正規表達式，更容錯地匹配各部分
    pattern = re.compile(
        r'\((?P<answer_letter>\w)\)\s*'                 # 答案字母與空格
        r'(?P<question_number>\d+)\.\s*'                # 題號與點
        r'(?P<question_text>.*?)\s*'                    # 題目文本，非貪婪匹配
        r'(?P<source_info>\(\*{1,2}.*?\)| \(時事\)| \(時事.*?\))?' # 題目標記，這裡更具容錯性
        r'\s*(?P<options_text>(?:\(.\).*?){4})?'        # 選項文本，非貪婪匹配4個
        r'(?=\s*\(\w\)\s*\d|\s*\Z)',
        re.S | re.M
    )
    
    matches = re.finditer(pattern, processed_text)

    # 用於從選項文字中提取選項列表的正規表達式
    option_pattern = re.compile(r'\([A-D]\)\s*(.*?)(?=\s*\([A-D]\)|\s*\Z)')

    for match in matches:
        try:
            answer_letter = match.group('answer_letter').strip()
            question_number = match.group('question_number').strip()
            question_text = match.group('question_text').strip()
            source_info_str = match.group('source_info').strip().strip('()') if match.group('source_info') else ""
            options_text = match.group('options_text').strip() if match.group('options_text') else ""

            # 提取選項列表
            options_list = [opt.strip() for opt in re.findall(option_pattern, options_text)]

            # 確保選項列表有四個，否則視為解析失敗
            if len(options_list) != 4:
                print(f"警告：在 {exam_date} 的第 {question_number} 題選項不完整，可能為解析錯誤。已跳過該題。")
                continue
            
            # 根據答案字母找到完整的答案文字
            answer_text = "無答案"
            if len(options_list) >= ord(answer_letter.upper()) - ord('A') + 1:
                answer_index = ord(answer_letter.upper()) - ord('A')
                answer_text = options_list[answer_index]

            # 根據標記處理出處和頁次
            book_source = "無"
            page_number = "無"

            if '**' in source_info_str:
                book_source = "常用醫護術語"
                page_match = re.search(r'\*\*(.*)', source_info_str)
                if page_match:
                    page_number = page_match.group(1).strip()
            elif '*' in source_info_str:
                book_source = "醫學資訊管理學"
                page_match = re.search(r'\*(.*)', source_info_str)
                if page_match:
                    page_number = page_match.group(1).strip()
            
            if '時事' in source_info_str:
                if book_source != "無":
                    book_source += "、時事"
                else:
                    book_source = "時事"

            questions.append({
                "題目": question_text,
                "選項": options_list,
                "考試時間": exam_date,
                "出自哪一本書": book_source,
                "頁次": page_number,
                "答案": answer_text,
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
