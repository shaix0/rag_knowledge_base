import os
import sys
import csv
import json # 新增 json 模組以解析 AI 輸出的結構化數據
from mistralai import Mistral # 確保您已安裝 'mistralai' 庫
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# --- 配置區塊 ---
INGREDIENT_CSV_DELIMITER = ',' 
MISTRAL_MODEL = "mistral-small-latest" 

# --- 營養素欄位定義 (與輸出 CSV 格式一致) ---
DATA_FIELDNAMES = ["dish_id", "calories", "mass", "fat", "carb", "protein"]

# --- 數據結構類型 ---
# { "ingredient_name": {"cal/g": 0.98, "fat(g)": 0.043, ...} }
NutrientData = Dict[str, float]
IngredientDB = Dict[str, NutrientData]
# AI 輸出的結構化數據類型
EstimatedFood = Dict[str, Any] 


def SimpleSingularize(word: str) -> str:
    """
    增強版：嘗試將一個單詞轉換為其單數形式。
    專門處理常見的食物單複數變化，以提高與數據庫的匹配成功率。
    """
    word = word.lower().strip()
    
    # 處理特殊或不規則名詞 (在食物中常見)
    irregular_map = {
        'rice': 'rice', 
        'fish': 'fish',
        'sushi': 'sushi', 
        'potatoes': 'potato',
        'tomatoes': 'tomato',
        'berries': 'berry',
        'cookies': 'cookie',
        'fries': 'fry', # 例如: french fries -> french fry
        'peppers': 'pepper'
    }
    
    if word in irregular_map:
        return irregular_map[word]
    
    # 處理 '-ies' 結尾: e.g., 'strawberries' -> 'strawberry'
    if word.endswith('ies'):
        return word[:-3] + 'y'
    
    # 處理 '-es' 結尾: e.g., 'bunches' -> 'bunch'
    if word.endswith('es') and len(word) > 3:
        # 排除像 'cheese' 這種單數本身就以 'es' 結尾的單詞
        if word[:-2] not in irregular_map: 
             return word[:-2]
    
    # 處理大部分 '-s' 結尾: e.g., 'apples' -> 'apple'
    # 排除單數名詞如 'cheese' 或以 'ss' 結尾的名詞
    if word.endswith('s') and len(word) > 2 and not word.endswith(('ss', 'us', 'is')):
        return word[:-1]
    
    return word


def ReadIngredientMetadata(filepath: str) -> IngredientDB:
    """讀取食材元數據 CSV 檔案，並建立一個以食材名稱為鍵的字典。"""
    if not os.path.exists(filepath):
        print(f"錯誤: 食材元數據檔案未找到: {filepath}")
        return {}
    
    ingredient_db: IngredientDB = {}
    
    expected_headers = ["ingr", "id", "cal/g", "fat(g)", "carb(g)", "protein(g)"]
    
    with open(filepath, "r", newline='', encoding='utf-8-sig') as f_in:
        reader = csv.reader(f_in, delimiter=INGREDIENT_CSV_DELIMITER)
        
        try:
            headers = next(reader)
            if not all(h in headers for h in expected_headers):
                 print(f"警告: 標頭不符合預期: {headers}")
            
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
                    "mass": 1.0 
                }
            except (IndexError, ValueError) as e:
                print(f"警告: 處理食材數據時出錯，跳過該行: {row} ({e})")
                continue

    return ingredient_db


def GetIdentifiedIngredients(image_url: str, api_key: str, valid_ingredients: List[str]) -> List[EstimatedFood]:
    """呼叫 Mistral API 進行圖像辨識，並返回包含食材名稱和估算質量的列表。"""
    print(f"正在呼叫 Mistral 模型 ({MISTRAL_MODEL}) 辨識圖片和估算重量: {image_url}...")
    
    client = Mistral(api_key=api_key)
    
    # 將所有有效食材名稱合併成一個易讀的字串，用於提示詞
    ingredient_list_str = ", ".join(sorted(valid_ingredients))
    
    # 增強的系統提示詞，明確指示模型參考列表並以 JSON 格式輸出名稱和估算質量 (mass_g)
    system_prompt = (
        "You are an expert food identification and serving estimation AI. "
        "Your task is to identify as many different food ingredients as possible in the image and estimate the mass (in grams) "
        "of a typical serving of each item based on common knowledge (e.g., one whole apple is 182g, "
        "one banana is 118g, a standard slice of bread is 28g). "
        "Your priority is to match the ingredient name to the following list: "
        f"[{ingredient_list_str}]. Please prioritize singular and plural forms from the list. "
        "If a food is not in the list, use the closest common name. "
        "You MUST respond ONLY with a single JSON array of objects. "
        "Each object MUST have two keys: 'name' (string, the food name) and 'mass_g' (float, the estimated mass in grams). "
        "DO NOT include any extra text, explanation, or markdown formatting outside the JSON array."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please identify the ingredients and estimate their typical mass in grams for this image:"},
                {"type": "image_url", "image_url": image_url}
            ]
        }
    ]
    
    try:
        chat_response = client.chat.complete(
            model=MISTRAL_MODEL,
            messages=messages
        )
        
        raw_output_content = chat_response.choices[0].message.content
        
        # 嘗試清理任何可能的 markdown 標記 (```json) 或不必要的文字
        if isinstance(raw_output_content, str):
            if raw_output_content.startswith("```json"):
                raw_output_content = raw_output_content.replace("```json", "").replace("```", "").strip()
        
        identified_foods_with_mass = []
        try:
            # 解析 JSON 陣列 [{"name": "...", "mass_g": 0.0}, ...]
            parsed_foods = json.loads(raw_output_content)
        except json.JSONDecodeError as e:
            print(f"錯誤: 無法解析 Mistral API 輸出的 JSON 格式: {e}")
            print(f"原始輸出: {raw_output_content[:200]}...")
            return []

        # 驗證結構並標準化
        for item in parsed_foods:
            if isinstance(item, dict) and 'name' in item and 'mass_g' in item:
                # Standardize name and ensure mass is a float
                name = str(item['name']).strip().lower()
                mass = float(item['mass_g']) if item['mass_g'] is not None else 0.0
                if name and mass > 0:
                    identified_foods_with_mass.append({"name": name, "mass_g": mass})
                
        return identified_foods_with_mass
        
    except Exception as e:
        print(f"Mistral API 呼叫失敗: {e}")
        return []


def CalculateNutrition(identified_foods_with_mass: List[EstimatedFood], ingredient_db: IngredientDB) -> Tuple[Dict[str, float], List[str]]:
    """計算匹配成功的食材的總營養素，並使用 AI 估算的質量進行計算。
    
    此函數已強化匹配邏輯：先嘗試直接匹配，失敗後再嘗試單數形式。
    """
    total_nutrition = defaultdict(float)
    matched_ingredients = []
    
    print("\n--- 營養素計算步驟 ---")
    
    for item in identified_foods_with_mass:
        food = item.get('name', '')
        estimated_mass = item.get('mass_g', 0.0)
        
        if not food or estimated_mass <= 0:
             print(f"  [跳過] 食材: {food.capitalize()} (質量無效: {estimated_mass:.1f}g)")
             continue
             
        normalized_food = food.strip().lower() 
        match_name = None 
        
        # 1. 嘗試直接匹配 AI 輸出 (例如：如果 AI 輸出 'apples' 且 DB 也有 'apples')
        if normalized_food in ingredient_db:
            match_name = normalized_food
            
        # 2. 如果原始形式未匹配，則嘗試將 AI 輸出轉換為單數形式 (例如：AI 輸出 'apples', DB 只有 'apple')
        if match_name is None:
            singular_food = SimpleSingularize(normalized_food)
            
            # 只有在單數形式與原始形式不同時才進行檢查，避免重複
            if singular_food != normalized_food and singular_food in ingredient_db:
                match_name = singular_food
        
        # 3. 執行計算
        if match_name:
            # 記錄匹配到的數據庫名稱
            matched_ingredients.append(match_name)
            nutrients_per_g = ingredient_db[match_name]
            
            # 執行營養素計算 (使用 AI 估算的質量)
            total_nutrition['calories'] += nutrients_per_g['calories'] * estimated_mass
            total_nutrition['fat'] += nutrients_per_g['fat'] * estimated_mass
            total_nutrition['carb'] += nutrients_per_g['carb'] * estimated_mass
            total_nutrition['protein'] += nutrients_per_g['protein'] * estimated_mass
            total_nutrition['mass'] += estimated_mass # 總質量也累加
            
            print(f"  [匹配成功] 食材: {food.capitalize()} (數據庫名稱: {match_name.capitalize()}, 估算 {estimated_mass:.1f}g)")
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
        with open(output_path, 'w', newline='', encoding='utf-8') as f_out:
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
    
    dish_id = "dish_20251002"
    # 圖片 URL
    image_url = "https://raw.githubusercontent.com/google-research-datasets/Nutrition5k/refs/heads/main/res/example_plate.jpg"
    
    output_csv_path = os.path.join(
        os.path.dirname(__file__), 
        'information', 
        f'{dish_id}_prediction.csv'
    )
    
    ingredient_csv_path = os.path.join(
        os.path.dirname(__file__), 
        'information', 
        'nutrition5k_dataset_metadata_ingredients_metadata.csv'
    )
    
    # --- END: 硬編碼參數設定區塊 ---
    
    # 1. 載入食材數據庫並提取名稱列表
    ingredient_db = ReadIngredientMetadata(ingredient_csv_path)
    if not ingredient_db:
        print("無法載入食材數據庫，程式終止。")
        sys.exit(1)
        
    print(f"成功載入 {len(ingredient_db)} 種食材數據。")
    
    # 獲取所有有效的食材名稱，用於提示詞
    valid_ingredient_names = list(ingredient_db.keys())
    
    # 2. 圖像辨識 (包含數據庫名稱參考和重量估算)
    identified_foods_with_mass = GetIdentifiedIngredients(image_url, api_key, valid_ingredient_names)
    
    if not identified_foods_with_mass:
        print("AI 未能辨識出任何食材或估算質量，或 API 呼叫失敗。")
        sys.exit(0)
    
    # 輸出AI估算的清單，以更清晰的方式呈現
    print(f"\nAI 辨識出的原始食材與估算質量:")
    for item in identified_foods_with_mass:
        print(f"  - 名稱: {item.get('name', 'N/A').capitalize()}, 質量: {item.get('mass_g', 0.0):.1f}g")

    # 3. 營養素計算
    total_nutrition, matched_ingredients = CalculateNutrition(identified_foods_with_mass, ingredient_db)
    
    if not matched_ingredients:
        print("沒有任何 AI 辨識出的食材與您的 Nutrition5k 清單匹配，無法計算營養素。")
        sys.exit(0)

    # 4. 儲存結果
    WritePredictionCSV(dish_id, total_nutrition, output_csv_path)

if __name__ == "__main__":
    main()
