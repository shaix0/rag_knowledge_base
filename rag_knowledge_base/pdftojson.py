import fitz  # PyMuPDF
import json
import re
import os

def parse_questions_from_text(text, exam_date, source_filename):
    """
    解析原始文本以提取題目、答案、選項、出處等資訊，並將元數據加入其中。
    """
    questions = []
    
    # 移除空格和換行
    processed_text = re.sub(r'\s+', '', text)
    print(processed_text)
    # 重新設計更強健的正規表達式，更容錯地匹配各部分
    pattern = re.compile(
        r'(?P<answer_letters>(?:\([A-D]\)(?:\s*or\s*\([A-D]\))*))\s*' # 提取答案字母，支援 "(A)or(C)" 等格式
        r'(?P<question_number>\d+)\.\s*'              # 題號與點
        r'(?P<question_text>.*?)\s*'                  # 題目文本，非貪婪匹配
        r'(?P<source_info>\(\*{1,2}.*?\)| \(時事\)| \(時事.*?\))?' # 來源，這裡更具容錯性
        r'\s*(?P<options_text>(?:\(.\).*?){4})?'      # 選項文本，匹配4個
        r'(?=\s*\(\w\)\s*\d|\s*\Z)',
        #re.S | re.M
    )
    
    matches = re.finditer(pattern, processed_text)

    # 分割選項
    option_pattern = re.compile(r'\([A-D]\)(.*?)(?=\([A-D]\)|$)')

    for match in matches:
        try:
            answer_letters_str = match.group('answer_letters').strip()
            answer_letters = re.findall(r'\(([A-D])\)', answer_letters_str)  # 直接取得所有字母
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
            
            # 根據答案字母字串找到完整的答案文字
            # 處理多個答案，例如 "A or C"
            answer_text = "無答案"
            correct_answers_text = []

            for letter in answer_letters:
                if len(options_list) >= ord(letter.upper()) - ord('A') + 1:
                    answer_index = ord(letter.upper()) - ord('A')
                    correct_answers_text.append(options_list[answer_index])
            
            if correct_answers_text:
                answer_text = " 或 ".join(correct_answers_text)

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
                "答案": answer_text,
                "來源書籍": book_source,
                "頁次": page_number,
                #"考試時間": exam_date,
                "來源檔案": source_filename
            })
        except Exception as e:
            print(f"解析題目時發生錯誤: {e}. 原始匹配文字: {match.group(0)}")
            continue

    return questions

def merge_questions(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    merged = {} # dict

    for q in data:
        key = (q["題目"], q["答案"]) # 以題目和答案作為key，這裡的資料類型是tuple，是不可變的
        if key not in merged:
            merged[key] = {
                "題目": q["題目"],
                "選項": q["選項"],
                "答案": q["答案"],
                "來源書籍": q.get("來源書籍", ""),
                "頁次": q.get("頁次", ""),
                "來源檔案": set([q.get("來源檔案")]) if q.get("來源檔案") else set(),
            }
        else:
            # 出現重複題目，合併題目只保留來源檔案。
            if q.get("來源檔案"):
                if isinstance(q["來源檔案"], list): 
                    merged[key]["來源檔案"].update(q["來源檔案"])
                else:
                    merged[key]["來源檔案"].add(q["來源檔案"])

    # 把set轉回list
    merged_list = []
    for v in merged.values():
        v["來源檔案"] = list(v["來源檔案"])
        merged_list.append(v)

    # 重新寫入json
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(merged_list, f, ensure_ascii=False, indent=4)

    print(f"合併完成，共 {len(merged_list)} 題，已輸出至 {json_path}")

# 獲取資料夾中所有以 .pdf 結尾的檔案
pdf_folder_path = os.path.join(os.path.dirname(__file__), 'information', '醫學資訊管理師')

try:
    pdf_files = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]
except FileNotFoundError:
    print(f"錯誤: 找不到指定的資料夾 '{pdf_folder_path}'")
    pdf_files = []

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
        #print(text)
        print(f"成功從 {filename} 提取文本。")

        # 使用解析函式處理文本
        questions_from_file = parse_questions_from_text(text, exam_date, filename)
        all_extracted_questions.extend(questions_from_file)

    except Exception as e:
        print(f"從 {filename} 提取文本或解析時發生錯誤: {e}")
        continue

# 將所有整理好的題目儲存為 JSON 檔案
output_folder = os.path.join(os.path.dirname(__file__), 'information', '醫學資訊管理師')  # 指定輸出資料夾
# 確保輸出資料夾存在
os.makedirs(output_folder, exist_ok=True)
output_filename = os.path.join(output_folder, "all_questions.json")  # 將檔案儲存到指定資料夾

with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(all_extracted_questions, f, ensure_ascii=False, indent=4)

print(f"\n所有 {len(all_extracted_questions)} 個題目已成功整理並儲存至 {output_filename}")

merge_questions(output_filename)

'''
# --- LangChain 與 ChromaDB 整合部分 ---
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# 初始化 Ollama 模型
llm = OllamaLLM(model="mistral")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 2. 將 JSON 資料轉換為 LangChain Document 格式
documents = []
for item in all_extracted_questions:
    # 組合題目與選項，作為主要內容
    page_content = f"題目: {item['題目']}\n選項: {', '.join(item['選項'])}\n答案: {item['答案']}"
    
    # 將其他資訊作為元資料 (metadata)
    metadata = {
        "考試時間": item.get('考試時間', ''),
        "出自哪一本書": item.get('出自哪一本書', ''),
        "頁次": item.get('頁次', ''),
        "來源檔案": item.get('來源檔案', ''),
        # 這裡不處理 tags，因為這個腳本主要是解析和建立
    }
    
    # 建立 Document 物件並加入列表
    documents.append(Document(page_content=page_content, metadata=metadata))

if not documents:
    print("沒有可用的文件來建立向量資料庫。")
else:
    # 3. 建立向量資料庫 medical_questions
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="db",
        collection_name="medical_questions",
    )
    print("成功從 JSON 文件建立向量資料庫。")
'''

'''
def clear_database():
    """
    Clears the entire vector database by removing the 'db' directory.
    This action is irreversible and should be used with caution.
    """
    if os.path.exists(db_directory):
        print(f"正在清除資料庫： {db_directory}")
        shutil.rmtree(db_directory)
        print("資料庫已清除。")
    else:
        print("資料庫不存在，無需清除。")
'''
