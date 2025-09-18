'''import os
import json
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# === 1. 初始化模型與 embeddings ===
llm = OllamaLLM(model="mistral")
embeddings = OllamaEmbeddings(model="mistral")

# Chroma 資料庫目錄
DB_DIR = "chroma_db"

# === 2. 定義檔案路徑 ===
knowledge_base_directory = os.path.join(os.path.dirname(__file__), 'rag_knowledge_base', 'information', '醫學資訊管理師')
input_json_path = os.path.join(knowledge_base_directory, "all_questions.txt")
output_json_path = os.path.join(knowledge_base_directory, "all_questions.json") #input_json_path  # 覆寫同一份

# === 3. 建立 / 載入 Chroma 向量資料庫 ===
def build_or_load_vectorstore():
    if os.path.exists(DB_DIR):
        print("載入既有 Chroma 資料庫...")
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    else:
        print("建立新的 Chroma 資料庫...")
        # 讀取 JSON
        with open(input_json_path, "r", encoding="utf-8-sig") as f:
            questions_data = json.load(f)
        
        docs = []
        for q in questions_data:
            text = f"題目: {q.get('題目', '')}\n答案: {q.get('答案', '')}"
            docs.append(text)

        # 切分文本（避免太長）
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = splitter.create_documents(docs)

        vectordb = Chroma.from_documents(split_docs, embeddings, persist_directory=DB_DIR)
        return vectordb

# === 4. 定義生成標籤的方法 ===
def generate_tags_for_questions():
    vectordb = build_or_load_vectorstore()

    # 讀取題目資料
    with open(input_json_path, "r", encoding="utf-8-sig") as f:
        questions_data = json.load(f)

    tag_prompt_template = ChatPromptTemplate.from_messages([
        ("system", "你是一個標籤生成助手。根據提供的題目和答案，生成1到3個最相關的繁體中文標籤。標籤應該是專有名詞或短語，如果有英文專有名詞應以其中文名為主。不要有任何解釋或額外文字。只回傳以逗號分隔的標籤，例如：資料庫,醫療法規,醫學倫理"),
        ("user", "題目: {題目}\n答案: {答案}\n相關題目:\n{相關題目}")
    ])

    for idx, q in enumerate(questions_data, start=1):
        query_text = f"{q.get('題目', '')} {q.get('答案', '')}"

        # === 檢索相關題目 ===
        results = vectordb.similarity_search(query_text, k=3)
        related = "\n".join([doc.page_content for doc in results])

        # === 呼叫 LLM 產生標籤 ===
        try:
            prompt = tag_prompt_template.format(
                題目=q.get("題目", ""),
                答案=q.get("答案", ""),
                相關題目=related
            )
            response = llm.invoke(prompt)
            tags = [t.strip() for t in response.split(",") if t.strip()]
            q["標籤"] = tags
            print(f"[{idx}] 已完成: {q['題目']} → {tags}")
        except Exception as e:
            print(f"[{idx}] 出錯: {e}")
            q["標籤"] = []

    # === 寫回 JSON ===
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(questions_data, f, ensure_ascii=False, indent=4)
    print(f"✅ 標籤已更新並儲存至 {output_json_path}")

if __name__ == "__main__":
    generate_tags_for_questions()

'''

from langchain_community.llms import Ollama
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os

llm = Ollama(model="mistral")

loader = TextLoader('rag_knowledge_base/information/醫學資訊管理師/all_questions.txt')

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=20,
    chunk_overlap=5,
    separators=[" ", ",", "\n"],
)

splited_docs = text_splitter.split_documents(loader.load())

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_db = Chroma.from_documents(
    documents=splited_docs,
    embedding=embeddings,
    persist_directory="db",
    collection_name="about",
)

retriever = vector_db.as_retriever(search_kwargs={"k": 3})

system_prompt = "你是一個標籤生成助手。根據提供的題目和答案，生成1到3個最相關的繁體中文標籤。標籤應該是專有名詞或短語，如果有英文專有名詞應以其中文名為主。不要有任何解釋或額外文字，不要有簡體中文字。"
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "問題: {input}"),
    ]
)

document_chain = create_stuff_documents_chain(llm, prompt_template)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

context = []
input_text = input("您想問什麼問題？\n>>> ")

while input_text.lower() != "bye":
    response = retrieval_chain.invoke({"input": input_text, "context": context})
    context = response["context"]

    print(response["answer"])

    input_text = input(">>> ")