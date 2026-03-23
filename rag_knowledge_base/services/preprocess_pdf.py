import fitz  # PyMuPDF
import json
import re
import os

import os
import re
import fitz


def parse_pdf_content():
    pdf_path = r"C:\Users\User\source\repos\rag_knowledge_base-1\rag_knowledge_base\information\醫學資訊管理師\2024年4月.pdf"
    filename = os.path.basename(pdf_path)
    text = ""

    try:
        with fitz.open(pdf_path) as pdf:
            for page in pdf:
                text += page.get_text()

        print(f"成功從 {filename} 提取文本。")

        questions_from_file = parse_questions_from_text(text)

        print(questions_from_file[:2])  # 預覽

    except Exception as e:
        print(f"從 {filename} 提取文本或解析時發生錯誤: {e}")


def parse_questions_from_text(text):

    # 保留換行，只清理多餘空白
    text = re.sub(r'[ \t]+', ' ', text)
    # 題目切割 pattern
    question_pattern = re.compile(
        r"((?:\([A-D]\)\s*(?:or\s*)?)+\s*\d+\..*?)(?=(?:\([A-D]\)\s*(?:or\s*)?)+\s*\d+\.|\Z)",
        re.S
    )
    questions = question_pattern.findall(text)

    data = []

    for q_text in questions:

        # 取得題號
        number_match = re.search(r"(\d+)\.", q_text)
        number = int(number_match.group(1)) if number_match else None

        # 抓題目文字（去掉答案區）
        q_body = re.split(r"\b\d+\.", q_text, 1)
        question_body = q_body[1] if len(q_body) > 1 else q_text
        question_text = re.split(r"\([A-D]\)", question_body)[0].strip()
        question_text = re.sub(r'\s+', '', question_text)

        # 抓來源，假設來源在題目文字中以 "來源: XXX" 的格式出現 
        source_match = re.search(r"\((\*+|時事)(\d+)?\)", q_text) 
        source = None
        page_number = None

        if source_match:
            source_type = source_match.group(1)
            page_str = source_match.group(2)

            # 替換來源名稱
            source_map = {
                "*": "教材A",
                "**": "教材B",
                "時事": "時事"
            }

        source = source_map.get(source_type, source_type)

        if page_str:
            page_number = int(page_str)

        # 抓選項
        # 找第一個選項位置
        option_start = re.search(r"\([A-D]\)", q_text)

        options = []
        option_map = {}

        if option_start:

            option_text = q_text[option_start.start():]

            option_pattern = re.compile(r"\(([A-D])\)\s*([^()]+)")
            option_matches = option_pattern.findall(option_text)

            for letter, content in option_matches:
                content = content.strip()
                options.append(content)
                option_map[letter] = content

        if option_start:
            question_text = q_text[:option_start.start()]
        else:
            question_text = q_text

        question_text = re.sub(r"^\d+\.\s*", "", question_text).strip()

        if option_start:
            question_text = q_text[number_match.end():option_start.start()].strip()
        else:
            question_text = q_text[number_match.end():].strip()

        # 抓答案
        answer_part = q_text.split(f"{number}.")[0]
        answer_letters = re.findall(r"\(([A-D])\)", answer_part)        # 把答案轉成文字
        answers = []
        for letter in answer_letters:
            if letter in option_map:
                answers.append(option_map[letter])

        # 抓說明
        exp_match = re.search(r"※\s*說明[:：]\s*(.*)", q_text)
        explanation = exp_match.group(1).strip() if exp_match else None

        data.append({
            "number": number,
            "question_text": question_text,
            "options": options,
            "answers": answers,
            "explanation": explanation,
            "source": source,
            "page_number": page_number
        })
    print(data[:2]) # 預覽前2題
    return data

parse_pdf_content()