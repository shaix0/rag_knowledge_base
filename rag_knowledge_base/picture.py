'''import os
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
model = "magistral-small-latest"
client = Mistral(api_key=api_key)

# Define the messages for the chat
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What's in this image?"
            },
            {
                "type": "image_url",
                "image_url": "https://i.epochtimes.com/assets/uploads/2019/04/shutterstock_295189109.jpg"
            }
        ]
    }
]

# Get the chat response
chat_response = client.chat.complete(
    model=model,
    messages=messages
)

# Print the content of the response
print(chat_response.choices[0].message.content)
'''

import os
import sys
import csv
from mistralai import Mistral # 確保您已安裝 'mistralai' 庫
from typing import Dict, List, Tuple
from collections import defaultdict

# --- 配置區塊 ---
# 假設您的 ingredient_metadata 檔案使用逗號分隔
INGREDIENT_CSV_DELIMITER = ',' 
DEFAULT_INGREDIENT_MASS_G = 100.0 # 關鍵假設：如果 AI 只辨識出名稱，我們假設每個食材都是 100 克
MISTRAL_MODEL = "mistral-small-latest" # 選擇支持多模態輸入的 Mistral 模型

# --- 營養素欄位定義 (與輸出 CSV 格式一致) ---
DATA_FIELDNAMES = ["dish_id", "calories", "mass", "fat", "carb", "protein"]

# --- 數據結構類型 ---
# { "ingredient_name": {"cal/g": 0.98, "fat(g)": 0.043, ...} }
NutrientData = Dict[str, float]
IngredientDB = Dict[str, NutrientData]


def ReadIngredientMetadata(filepath: str) -> IngredientDB:
    """讀取食材元數據 CSV 檔案，並建立一個以食材名稱為鍵的字典。"""
    if not os.path.exists(filepath):
        print(f"錯誤: 食材元數據檔案未找到: {filepath}")
        return {}
    
    ingredient_db: IngredientDB = {}
    
    # 預期的 CSV 標頭：ingr, id, cal/g, fat(g), carb(g), protein(g)
    expected_headers = ["ingr", "id", "cal/g", "fat(g)", "carb(g)", "protein(g)"]
    
    with open(filepath, "r", newline='', encoding='utf-8-sig') as f_in:
        reader = csv.reader(f_in, delimiter=INGREDIENT_CSV_DELIMITER)
        
        try:
            headers = next(reader)
            # 檢查標頭是否匹配 (跳過 id 欄位)
            if not all(h in headers for h in expected_headers):
                 print(f"警告: 標頭不符合預期: {headers}")
            
            # 找到關鍵營養素的索引
            ingr_idx = headers.index("ingr")
            cal_idx = headers.index("cal/g")
            fat_idx = headers.index("fat(g)")
            carb_idx = headers.index("carb(g)")
            protein_idx = headers.index("protein(g)")
            
        except StopIteration:
            print("錯誤: 食材檔案是空的或缺少標頭。")
            return {}
        except ValueError as e:
            print(f"錯誤: 食材檔案缺少必要的欄位標頭: {e}")
            return {}

        for row in reader:
            try:
                ingr_name = row[ingr_idx].strip().lower() # 標準化為小寫
                if not ingr_name: continue

                # 存儲每克的營養素數值
                ingredient_db[ingr_name] = {
                    "calories": float(row[cal_idx]),
                    "fat": float(row[fat_idx]),
                    "carb": float(row[carb_idx]),
                    "protein": float(row[protein_idx]),
                    "mass": 1.0 # 假設 mass_per_mass 單位為 1
                }
            except (IndexError, ValueError) as e:
                print(f"警告: 處理食材數據時出錯，跳過該行: {row} ({e})")
                continue

    return ingredient_db


def GetIdentifiedIngredients(image_url: str, api_key: str) -> List[str]:
    """呼叫 Mistral API 進行圖像辨識，並返回食材列表。"""
    print(f"正在呼叫 Mistral 模型 ({MISTRAL_MODEL}) 辨識圖片: {image_url}...")
    
    client = Mistral(api_key=api_key)
    
    # 優化提示詞，明確要求模型只輸出逗號分隔的食物列表
    system_prompt = (
        "You are an expert food identification AI. Identify all distinct food ingredients "
        "in the image and list them. List only the common name of the food, separated by commas. "
        "DO NOT include any explanation or extra text. Only the comma-separated list."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please identify the ingredients in this image:"},
                {"type": "image_url", "image_url": image_url}
            ]
        }
    ]
    
    try:
        chat_response = client.chat.complete(
            model=MISTRAL_MODEL,
            messages=messages
        )
        raw_output = chat_response.choices[0].message.content.lower().strip()
        
        # 解析輸出: 將逗號分隔的字串轉換為列表
        identified_foods = [food.strip() for food in raw_output.split(',') if food.strip()]
        return identified_foods
        
    except Exception as e:
        print(f"Mistral API 呼叫失敗: {e}")
        return []


def CalculateNutrition(identified_foods: List[str], ingredient_db: IngredientDB) -> Tuple[Dict[str, float], List[str]]:
    """計算匹配成功的食材的總營養素。"""
    total_nutrition = defaultdict(float)
    matched_ingredients = []
    
    print("\n--- 營養素計算步驟 ---")
    
    for food in identified_foods:
        # 進行比對與篩選
        if food in ingredient_db:
            matched_ingredients.append(food)
            nutrients_per_g = ingredient_db[food]
            
            # 執行營養素計算 (假設每個食材都是 100g)
            total_nutrition['calories'] += nutrients_per_g['calories'] * DEFAULT_INGREDIENT_MASS_G
            total_nutrition['fat'] += nutrients_per_g['fat'] * DEFAULT_INGREDIENT_MASS_G
            total_nutrition['carb'] += nutrients_per_g['carb'] * DEFAULT_INGREDIENT_MASS_G
            total_nutrition['protein'] += nutrients_per_g['protein'] * DEFAULT_INGREDIENT_MASS_G
            total_nutrition['mass'] += DEFAULT_INGREDIENT_MASS_G # 總質量也累加
            
            print(f"  [匹配成功] 食材: {food.capitalize()} (假設 {DEFAULT_INGREDIENT_MASS_G:.1f}g)")
        else:
            print(f"  [未匹配] 食材: {food.capitalize()} (不在 Nutrition5k 清單中)")
            
    # 如果沒有匹配的食材，確保所有數值為 0
    for field in DATA_FIELDNAMES[1:]:
        if field not in total_nutrition:
            total_nutrition[field] = 0.0
            
    return dict(total_nutrition), matched_ingredients


def WritePredictionCSV(dish_id: str, nutrition_data: Dict[str, float], output_path: str):
    """將計算結果寫入符合 Nutrition5k 預測格式的 CSV 檔案。"""
    
    # 創建輸出目錄 (如果不存在)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f_out:
            writer = csv.writer(f_out, delimiter=',')
            
            # 寫入標頭行
            writer.writerow(DATA_FIELDNAMES)
            
            # 寫入數據行 (確保順序與 DATA_FIELDNAMES 一致)
            data_row = [
                dish_id,
                f"{nutrition_data.get('calories', 0):.2f}",
                f"{nutrition_data.get('mass', 0):.2f}",
                f"{nutrition_data.get('fat', 0):.2f}",
                f"{nutrition_data.get('carb', 0):.2f}",
                f"{nutrition_data.get('protein', 0):.2f}"
            ]
            writer.writerow(data_row)
            
        print(f"\n--- 結果儲存成功 ---")
        print(f"預測結果已儲存至: {output_path}")
        print(f"計算結果: {data_row}")
        
    except Exception as e:
        print(f"寫入 CSV 檔案時出錯: {e}")


def main():
    """主執行函數，處理參數設定和整個工作流程。"""
    
    # 檢查 MISTRAL_API_KEY 環境變數
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("錯誤: 請設定環境變數 MISTRAL_API_KEY。")
        sys.exit(1)

    # --- START: 硬編碼參數設定區塊 (請直接修改這裡的值) ---
    
    # 1. 設置要處理的菜餚 ID (任意唯一字串)
    dish_id = "dish_20251002"
    
    # 2. 設置要分析的圖片 URL (這是您提供的範例圖片)
    image_url = "https://tokyo-kitchen.icook.network/uploads/recipe/cover/372355/445748b0bf0f2991.jpg"
    
    # 3. 設置輸出 CSV 文件的路徑 (將存放在腳本目錄下的 output_predictions 資料夾)
    output_csv_path = os.path.join(
        os.path.dirname(__file__), 
        'information', 
        f'{dish_id}_prediction.csv'
    )
    
    # 4. 設定食材 CSV 路徑為腳本目錄下的 'information/ingredient_metadata.csv' (此路徑已固定)
    ingredient_csv_path = os.path.join(
        os.path.dirname(__file__), 
        'information', 
        'nutrition5k_dataset_metadata_ingredients_metadata.csv'
    )
    
    # --- END: 硬編碼參數設定區塊 ---
    
    # 1. 載入食材數據庫
    ingredient_db = ReadIngredientMetadata(ingredient_csv_path)
    if not ingredient_db:
        print("無法載入食材數據庫，程式終止。")
        sys.exit(1)
        
    print(f"成功載入 {len(ingredient_db)} 種食材數據。")
    
    # 2. 圖像辨識
    identified_foods = GetIdentifiedIngredients(image_url, api_key)
    
    if not identified_foods:
        print("AI 未能辨識出任何食材，或 API 呼叫失敗。")
        sys.exit(0)
    
    print(f"\nAI 辨識出的原始食材: {identified_foods}")

    # 3. 營養素計算 (包含數據庫比對與篩選)
    total_nutrition, matched_ingredients = CalculateNutrition(identified_foods, ingredient_db)
    
    if not matched_ingredients:
        print("沒有任何 AI 辨識出的食材與您的 Nutrition5k 清單匹配，無法計算營養素。")
        sys.exit(0)

    # 4. 儲存結果
    WritePredictionCSV(dish_id, total_nutrition, output_csv_path)

if __name__ == "__main__":
    main()
