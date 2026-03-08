import fitz  # PyMuPDF
import json
import re
import os

def parse_pdf_content():
    pdf_path = os.path.join(os.path.dirname(__file__), 'information', '醫學資訊管理師', '2017年4月.pdf')
    filename = os.path.basename(pdf_path)
    text = []

    try:
        with fitz.open(pdf_path) as pdf:
            for page in pdf:
                text += page.get_text()
        #print(text)
        print(f"成功從 {filename} 提取文本。")

        # 使用解析函式處理文本
        questions_from_file = parse_questions_from_text(text, filename)
        all_extracted_questions.extend(questions_from_file)

    except Exception as e:
        print(f"從 {filename} 提取文本或解析時發生錯誤: {e}")

def parse_questions_from_text(text, filename):
    # 題目 pattern: 題號 + 題目文字 + (可能的頁碼)
    question_pattern = re.compile(r"(\(\w\)\s*\d+\..*?)(?=(\(\w\)\s*\d+\.|\Z))", re.S)
    questions = question_pattern.findall(text)

    data = []

    for q_tuple in questions:
        q_text = q_tuple[0].strip()
    
        # 取得題號
        number_match = re.search(r"\d+", q_text)
        number = int(number_match.group()) if number_match else None
    
        # 抓選項
        options_pattern = re.compile(r"\(([A-D])\)\s*(.*?)\s*(?=\([A-D]\)|$)", re.S)
        options_matches = options_pattern.findall(q_text)
    
        options = {opt: text.strip() for opt, text in options_matches}
    
        # 抓答案，如果題目文字中有答案提示，可自行 regex 抓，暫時用空 list
        answers = []  
    
        # 抓頁碼
        page_match = re.findall(r"\*\*(\d+(?:,\s*\d+)*)\)|\*(\d+)\)", q_text)
        page_numbers = []
        for m in page_match:
            nums = m[0] or m[1]
            page_numbers.extend([int(n) for n in nums.replace(" ", "").split(",")])
    
        data.append({
            "number": number,
            "question_text": q_text,
            "options": options,
            "answers": answers,
            "page_number": page_numbers
        })

    print(data[:2])  # 預覽前2題
