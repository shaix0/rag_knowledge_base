import os
import json
import time
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# 初始化 Ollama 模型
llm = OllamaLLM(model="mistral")

# 定義檔案路徑
knowledge_base_directory = os.path.join(os.path.dirname(__file__), 'rag_knowledge_base', 'information', '醫學資訊管理師')
input_json_path = os.path.join(knowledge_base_directory, "all_questions.json")
output_json_path = os.path.join(knowledge_base_directory, "all_questions.json")  # 將輸出檔案路徑設為與輸入相同

def generate_tags_for_questions():
    """
    從 JSON 檔案中讀取題目，並自動呼叫 LLM 產生標籤，然後將結果儲存回原本的 JSON 檔案。
    """
    try:
        # 1. 從 JSON 檔案中載入資料
        with open(input_json_path, 'r', encoding='utf-8-sig') as f:
            questions_data = json.load(f)
        print("成功載入 all_questions.json 檔案。")
    except FileNotFoundError:
        print(f"錯誤: 找不到知識庫檔案於 {input_json_path}")
        return
    except json.JSONDecodeError as e:
        print(f"解析 JSON 檔案時發生錯誤: {e}")
        return

    # 2. 定義標籤生成提示語
    tag_prompt_template = ChatPromptTemplate.from_messages([
        ("system", "你是一個標籤生成助手。根據提供的題目和答案，生成1到3個最相關的繁體中文標籤。標籤應該是專有名詞或短語，如果有英文專有名詞應以其中文名為主。不要有任何解釋或額外文字。只回傳以逗號分隔的標籤，例如：資料庫,醫療法規,醫學倫理"),
        ("user", "題目: {題目}\n答案: {答案}"),
    ])
    
    total_questions = len(questions_data)
    processed_count = 0
    
    # 3. 逐一處理每個題目並生成標籤
    print(f"開始為 {total_questions} 個題目生成標籤...")
    for question in questions_data:
        # 組合提示語中的變數，只使用題目和答案
        prompt_variables = {
            "題目": question.get("題目", ""),
            "答案": question.get("答案", "")
        }

        # 呼叫 LLM 產生標籤
        try:
            # 使用 LLM 的 invoke 方法來獲取響應
            response = llm.invoke(tag_prompt_template.format(**prompt_variables))
            
            # 將 LLM 的響應轉換為標籤列表
            tags = [tag.strip() for tag in response.split(',') if tag.strip()]
            question["標籤"] = tags  # 使用「標籤」作為鍵名

        except Exception as e:
            print(f"為題目 '{question['題目']}' 生成標籤時發生錯誤: {e}")
            question["標籤"] = []  # 出錯時給一個空標籤列表
        
        processed_count += 1
        print(f"已處理 {processed_count}/{total_questions} 個題目。")

    # 4. 將更新後的資料寫回原本的 JSON 檔案
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(questions_data, f, ensure_ascii=False, indent=4)
    
    print(f"\n所有題目已成功標註並儲存至 {output_json_path}")

# 執行標籤生成器
if __name__ == "__main__":
    generate_tags_for_questions()
