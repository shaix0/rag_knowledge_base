import getpass
import os
import json
import time
from mistralai import Mistral

api_key = os.environ.get("MISTRAL_API_KEY")

model = "mistral-small-latest"

# 步驟1: 讀取原始的 JSON 檔案
file_path = os.path.join(os.path.dirname(__file__), 'information', '醫學資訊管理師', 'all_questions.json')
with open(file_path, 'r', encoding='utf-8') as f:
    questions = json.load(f)

# 步驟2: 遍歷每個題目，生成標籤並將其加入
for item in questions:
    question = item['題目']
    answer = item['答案']

    prompt = f"""
        你是一個標籤生成助手。根據提供的題目和答案，請從提供的「題目」和「答案」中，提取1到3個最能代表主題的繁體中文關鍵字，標籤應為名詞或短語。
        範例：
        題目：白血病又稱血癌，是鄉土劇中常見的生病橋段。主因骨髓造血細胞產生不正常增生，進而影響骨髓造血功能的惡性疾病。『白血病』的英文是？
        答案：Leukemia
        標籤：白血病、血癌

        現在，請生成以下內容的關鍵字：
        題目：{question}
        答案：{answer}
    """
    
    # 步驟3: 呼叫 Mistral API 獲取關鍵字
    client = Mistral(api_key=api_key)
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    
    try:
        # 進行 API 呼叫
        chat_response = client.chat.complete(
            model=model,
            messages=messages,
        )
        
        # 處理回應
        keywords = chat_response.choices[0].message.content.strip().replace('標籤：', '')
        item['標籤'] = keywords.split('、')
        
        print(f"題目：{question}\n標籤：{item['標籤']}\n")

        # 步驟4: 加上延遲，避免超過速率限制        
    except Exception as e:
    # 判斷錯誤類型
        if "API error occurred: Status 429" in error_message.lower():
            print("偵測到速率限制錯誤，等待 1 秒後重試...")
            time.sleep(1) # 等待 1 秒
        else:
            print(f"在處理題目 '{question}' 時發生錯誤：{e}")
            continue # 繼續下一個題目  API error occurred: Status 429
        
# 步驟5: 將更新後的 JSON 資料寫回檔案
output_file_path = os.path.join(os.path.dirname(__file__), 'information', '醫學資訊管理師', 'all_questions_with_tags.json')
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(questions, f, ensure_ascii=False, indent=4)

print(f"\n所有標籤處理完成，已成功儲存至 {output_file_path}")


'''
import itertools
import json
import os
from mistralai import Mistral

file_path = 'all_questions_with_keywords.json'
with open(file_path, 'r', encoding='utf-8') as f:
    questions = json.load(f)


sentence_embeddings = [item['標籤'] for item in questions]
flat_sentence_embeddings = list(itertools.chain.from_iterable(sentence_embeddings))

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-embed"

client = Mistral(api_key=api_key)

embeddings_batch_response = client.embeddings.create(
    model=model,
    inputs=flat_sentence_embeddings,
)
'''