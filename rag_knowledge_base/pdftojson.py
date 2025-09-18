import fitz  # PyMuPDF
import json
import re
import os
import requests
import time

# Define the base path for both scripts to ensure consistency
base_path = os.path.join(os.path.dirname(__file__), 'information', '醫學資訊管理師')
pdf_folder_path = base_path
output_folder = base_path
output_filename = os.path.join(output_folder, "all_questions.json")

# Ollama API URL
OLLAMA_URL = "http://localhost:11434"
# Use nomic-embed-text for embeddings
EMBEDDING_MODEL = "nomic-embed-text"

def get_embedding(text):
    """
    使用 Ollama 模型將文字轉換為向量（embeddings）。
    """
    payload = {
        "model": EMBEDDING_MODEL,
        "prompt": text
    }
    
    # 指數退避重試機制
    retries = 5
    for i in range(retries):
        try:
            response = requests.post(f"{OLLAMA_URL}/api/embeddings", json=payload, timeout=60)
            response.raise_for_status()
            return response.json().get('embedding')
        except requests.exceptions.RequestException as e:
            print(f"嘗試 {i+1}/{retries} 次：呼叫 Ollama 產生向量時發生錯誤: {e}")
            time.sleep(2 ** i)  # 等待 2^i 秒後重試
    print("重試失敗，無法產生向量。")
    return None

def parse_questions_from_text(text, exam_date, source_filename):
    """
    Parses raw text to extract questions, answers, options, and source information,
    including metadata.
    """
    questions = []
    
    # 1. Text preprocessing: remove extra newlines, spaces, page numbers, headers/footers
    # Remove "Page X of Y" style page numbers
    processed_text = re.sub(r'第\d+頁,共\d+頁', '', text)
    # Remove titles like "CHIO", "PART I", or "Solution", and standalone numeric page numbers
    processed_text = re.sub(r'(?i)(?:CHIO|PART\s+I|Solution|Given\s+two.*|Example\s+\d+|Input|Output|Note:)\s*\n?', '', processed_text)
    processed_text = re.sub(r'\n\s*\d+\s*\n', '\n', processed_text)
    # Remove some common titles/notes
    processed_text = re.sub(r'社團法人台灣醫學資訊學會|醫學資訊管理師檢定考試試題|選擇題\d+題.*?請選擇一個最正確的答案。|\*表示.*?之頁次|\*\*表示.*?之頁次|獨立作業區', '', processed_text)
    
    # Replace newlines before options with a space
    processed_text = re.sub(r'\n(?=\s*\(\w\))', ' ', processed_text)
    # Remove extra spaces and newlines
    processed_text = re.sub(r'\s+', '', processed_text)
    
    # Redesigned robust regex to match all parts tolerantly
    pattern = re.compile(
        r'\((?P<answer_letters>[\w\sor]+?)\)\s*'      # Answer letters, supports "A or C" format
        r'(?P<question_number>\d+)\.\s*'              # Question number and dot
        r'(?P<question_text>.*?)\s*'                  # Question text, non-greedy match
        r'(?P<source_info>\(\*{1,2}.*?\)| \(時事\)| \(時事.*?\))?' # Source tag, more tolerant here
        r'\s*(?P<options_text>(?:\(.\).*?){4})?'      # Options text, non-greedy match for 4 items
        r'(?=\s*\(\w\)\s*\d|\s*\Z)',
        re.S | re.M
    )
    
    matches = re.finditer(pattern, processed_text)
    
    # Regex to extract options from the options text
    option_pattern = re.compile(r'\([A-D]\)\s*(.*?)(?=\s*\([A-D]\)|\s*\Z)')
    
    for match in matches:
        try:
            answer_letters_str = match.group('answer_letters').strip()
            question_number = match.group('question_number').strip()
            question_text = match.group('question_text').strip()
            source_info_str = match.group('source_info').strip().strip('()') if match.group('source_info') else ""
            options_text = match.group('options_text').strip() if match.group('options_text') else ""
            
            # Extract list of options
            options_list = [opt.strip() for opt in re.findall(option_pattern, options_text)]
            
            # Ensure there are four options, otherwise assume parsing failed and skip
            if len(options_list) != 4:
                print(f"警告：在 {exam_date} 的第 {question_number} 題選項不完整，可能為解析錯誤。已跳過該題。")
                continue
            
            # Find the full answer text based on the answer letters string
            # Handle multiple answers, e.g., "A or C"
            answer_text = "無答案"
            answer_letters = re.split(r'\s*or\s*', answer_letters_str)
            correct_answers_text = []
            
            for letter in answer_letters:
                if len(options_list) >= ord(letter.upper()) - ord('A') + 1:
                    answer_index = ord(letter.upper()) - ord('A')
                    correct_answers_text.append(options_list[answer_index])
            
            if correct_answers_text:
                answer_text = " 或 ".join(correct_answers_text)
            
            # Process source and page number based on the tags
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
                "來源書籍": book_source,
                "頁次": page_number,
                "答案": answer_text,
                "考試時間": exam_date,
                "來源檔案": source_filename
            })
        except Exception as e:
            print(f"解析題目時發生錯誤: {e}. 原始匹配文字: {match.group(0)}")
            continue
    
    return questions

# Get all PDF files in the specified folder
try:
    pdf_files = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]
except FileNotFoundError:
    print(f"錯誤: 找不到指定的資料夾 '{pdf_folder_path}'")
    pdf_files = []

all_extracted_questions = []

# Iterate through each PDF file
for pdf_path in pdf_files:
    filename = os.path.basename(pdf_path)
    # Extract exam date from the filename
    exam_date = os.path.splitext(filename)[0]
    
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        print(f"成功從 {filename} 提取文本。")
        
        # Process the text using the parsing function
        questions_from_file = parse_questions_from_text(text, exam_date, filename)
        all_extracted_questions.extend(questions_from_file)
    
    except Exception as e:
        print(f"從 {filename} 提取文本或解析時發生錯誤: {e}")
        continue

print(f"\n已從所有 PDF 檔案中解析出 {len(all_extracted_questions)} 個題目。")

# --- 產生向量嵌入並加入題目資料中 ---

print("\n正在為每個題目產生向量嵌入...")
for item in all_extracted_questions:
    # 組合題目與答案作為嵌入的輸入文本
    combined_text = f"題目: {item['題目']}\n答案: {item['答案']}"
    embedding = get_embedding(combined_text)
    if embedding:
        item['embedding'] = embedding
    else:
        print(f"警告: 無法為題目 '{item['題目'][:20]}...' 產生向量。")

# Save all processed questions to a single JSON file
os.makedirs(output_folder, exist_ok=True)

with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(all_extracted_questions, f, ensure_ascii=False, indent=4)

print(f"\n所有 {len(all_extracted_questions)} 個題目（包含向量）已成功整理並儲存至 {output_filename}")
print("現在您可以執行 langchain_from_json.py 來建立 ChromaDB 向量資料庫了。")
