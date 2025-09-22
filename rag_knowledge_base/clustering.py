'''from sklearn.cluster import KMeans
import pandas as pd
import os
import json
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-embed"

client = Mistral(api_key=api_key)

file_path = os.path.join(os.path.dirname(__file__), 'information', '醫學資訊管理師', 'tags.json')
with open(file_path, 'r', encoding='utf-8') as f:
    tags = json.load(f)

try:
    embeddings_batch_response = client.embeddings.create(
        model=model,
        inputs=tags,  # 直接傳入整個標籤列表
    )
    # 從回應中提取嵌入向量
    embeddings_list = [item.embedding for item in embeddings_batch_response.data]
#    embeddings_np = np.array(embeddings_list)
    print("成功生成標籤嵌入向量。")

except Exception as e:
    print(f"生成嵌入向量時發生錯誤: {e}")
    exit()

# 這裡開始需要修改 
model = KMeans(n_clusters=24, max_iter=1000)
cluster_labels = model.fit_predict(embeddings_list)

df = pd.DataFrame({'tag': tags, 'embedding': embeddings_list, 'cluster': cluster_labels})

#model.fit(df['embeddings'].to_list())
#df["cluster"] = model.labels_
#embeddings_list["cluster"] = model.labels_
#print(*df[df.cluster==23].text.head(3), sep='\n\n')

# 步驟 5: 檢視特定群集中的標籤
print("\n=== 各群集的標籤 ===")
for c in clusters:
    print(f"\n--- 群集 {c} ---")
    cluster_tags = df[df['cluster'] == c]['tag']
    for tag in cluster_tags:
        print(tag)'''
'''
from datetime import datetime
from flask import render_template, request
from rag_knowledge_base import app
import json
import os
import re
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

    tags_list = sorted(list(all_tags))
    output_folder = os.path.join(os.path.dirname(__file__), 'rag_knowledge_base', 'information', '醫學資訊管理師')  # 指定輸出資料夾
    output_filename = os.path.join(output_folder, "tags.json")  # 將檔案儲存到指定資料夾

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(tags_list, f, ensure_ascii=False, indent=4)

get_tag()'''




from datetime import datetime
from flask import render_template, request
from rag_knowledge_base import app
import json
import re
import os

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

knowledge_base_text = [f"題目: {item.get('題目', '')} 答案: {item.get('答案', '')}" for item in KNOWLEDGE_BASE]

from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-embed"
client = Mistral(api_key=api_key)

# 將所有知識庫文字轉換為向量
knowledge_base_vectors = client.embeddings.create(
    model="mistral-embed",
    inputs=knowledge_base_text
)
# 從 EmbeddingResponse 物件的 .data 屬性中提取嵌入向量列表
knowledge_base_vectors = [item.embedding for item in knowledge_base_vectors.data]

# 現在 knowledge_base_vectors 是一個列表，您可以安全地存取其元素
vector_dimension = len(knowledge_base_vectors[0])

# 建立一個 Faiss 索引
index = faiss.IndexFlatL2(vector_dimension)
index.add(np.array(knowledge_base_vectors).astype('float32'))