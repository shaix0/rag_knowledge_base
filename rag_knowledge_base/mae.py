"""Script to compute statistics on nutrition predictions using string inputs.

這個腳本接收 Ground Truth 和 Prediction 的 JSON 內容字串，計算絕對平均誤差 (MAE) 
和百分比平均誤差 (MAE %)，這些指標與 Nutrition5k 論文中用於評估模型的指標相似。
"""

import json
import statistics
import sys
import io 

DISH_ID_INDEX = 0
# 定義欄位名稱的順序，這必須與輸入的 JSON 陣列中的數據順序一致。
DATA_FIELDNAMES = ["dish_id", "calories", "mass", "fat", "carb", "protein"]


def ParseJsonData(json_content: str) -> dict:
  """解析 JSON 格式的字串數據，並以 dish_id (第一欄) 為鍵建立字典。
  
  預期 JSON 格式為:
  [ ["dish_id", "calories", "mass", "fat", "carb", "protein"], ...]
  """
  parsed_data = {}
  try:
    # 修正 1: 移除字串前後的空白和換行符，以確保 JSON 解析的正確性
    cleaned_content = json_content.strip()

    # 修正 2: 處理非標準 JSON 格式 (例如使用單引號 ' 代替標準的雙引號 ")
    # 注意: 這個替換是為了應對常見錯誤，但標準 JSON 應使用雙引號。
    standardized_content = cleaned_content.replace("'", '"')
    
    # 嘗試解析 JSON 字串，預期頂層是陣列
    data_list_of_lists = json.loads(standardized_content)
    
    if not isinstance(data_list_of_lists, list):
        print("JSON 解析錯誤: 預期頂層結構為列表 (Array)。")
        return {}

    expected_len = len(DATA_FIELDNAMES)
    
    for item_list in data_list_of_lists:
        # 驗證每個內部列表的結構
        if not isinstance(item_list, list) or len(item_list) != expected_len:
             print(f"警告: JSON 項目必須為包含 {expected_len} 個元素的列表，已跳過: {item_list}")
             continue
        
        # 取得菜餚 ID 作為字典的鍵
        dish_id = str(item_list[DISH_ID_INDEX]).strip()
        if not dish_id: continue
        
        # 將有序列表數據轉換為帶有欄位名稱的字典
        item_dict = {}
        for i, field_name in enumerate(DATA_FIELDNAMES):
            # 數據以字串形式儲存，留待計算時轉換為浮點數
            item_dict[field_name] = str(item_list[i])
            
        parsed_data[dish_id] = item_dict 
    
  except json.JSONDecodeError as e:
    print(f"JSON 解析錯誤: 請檢查輸入字串是否為有效的 JSON 格式: {e}")
    return {}
    
  return parsed_data

# --- 範例輸入數據 (請將您的真實 JSON 內容貼入這裡) ---

# 範例 Ground Truth 數據 (這是真實值)
# 格式: [ [dish_id, calories, mass, fat, carb, protein], ... ]
GROUNDTRUTH_JSON_CONTENT = """
[
  ['dish_20251002', '583.40', '481.00', '23.30', '52.40', '42.30']
]
"""

# 範例 Prediction 數據 (這是模型預測值)
PREDICTIONS_JSON_CONTENT = """
[
  ['dish_20251002', '622.30', '530.00', '17.25', '77.55', '38.97'],
  ['dish_20251002', '371.80', '410.00', '2.13', '77.75', '12.40'],
  ['dish_20251002', '371.80', '410.00', '2.13', '77.75', '12.40'],
  ['dish_20251002', '381.30', '440.00', '2.24', '79.67', '13.02'],
  ['dish_20251002', '498.30', '460.00', '10.34', '61.47', '40.04']
]
"""

# 輸出的 JSON 檔案路徑 (仍然需要一個路徑來儲存統計結果)
output_path = "output_statistics.json" 

# -----------------------------------------------------

# 1. 解析輸入字串
print("解析 Ground Truth JSON 字串...")
groundtruth_data = ParseJsonData(GROUNDTRUTH_JSON_CONTENT)
print("解析 Predictions JSON 字串...")
prediction_data = ParseJsonData(PREDICTIONS_JSON_CONTENT)

# 2. 數據處理與統計計算
groundtruth_values = {}
err_values = {}
output_stats = {}

for field in DATA_FIELDNAMES[1:]:
  groundtruth_values[field] = []
  err_values[field] = []

# 僅處理同時存在於 Ground Truth 和 Prediction 中的 dish_id
common_dish_ids = set(groundtruth_data.keys()) & set(prediction_data.keys())
    
print(f"找到 {len(common_dish_ids)} 個共同的菜餚 ID 進行比較。")


for dish_id in common_dish_ids:
  # 從數據中提取數值 (現在是字典，使用 key 存取)
  gt_dict = groundtruth_data[dish_id]
  pred_dict = prediction_data[dish_id]
  
  # 遍歷營養素欄位 (calories, mass, fat, carb, protein)
  for field_name in DATA_FIELDNAMES[1:]:
    try:
        # 使用 .get() 安全地從字典中提取數值
        # 由於數據是以字串形式儲存的，這裡將它們轉換為浮點數
        gt_value = float(gt_dict.get(field_name, 0.0))
        pred_value = float(pred_dict.get(field_name, 0.0))
        
        groundtruth_values[field_name].append(gt_value)
        err_values[field_name].append(abs(pred_value - gt_value))
    except (TypeError, ValueError):
        print(f"警告: 菜餚 {dish_id}, 欄位 {field_name} 的數值轉換失敗，跳過該數據點。")
        continue

# 3. 計算 MAE 和 MAE %
for field in DATA_FIELDNAMES[1:]:
    if groundtruth_values[field]:
        mean_err = statistics.mean(err_values[field])
        mean_gt = statistics.mean(groundtruth_values[field])
        
        output_stats[field + "_MAE"] = mean_err
        
        # 避免除以零
        if mean_gt != 0:
             output_stats[field + "_MAE_%"] = 100 * mean_err / mean_gt
        else:
             output_stats[field + "_MAE_%"] = 0.0
    else:
        output_stats[field + "_MAE"] = 0.0
        output_stats[field + "_MAE_%"] = 0.0

# 4. 寫入結果
try:
    # 註釋掉檔案寫入，直接在控制台輸出結果
    # with open(output_path, "w") as f_out:
    #   f_out.write(json.dumps(output_stats, indent=2))
      
    print(f"\n統計計算完成。")
    print("\n--- 計算結果摘要 (MAE) ---")
    for key, value in output_stats.items():
        # 僅顯示 MAE
        if key.endswith("MAE"):
            print(f"{key}: {value:.2f}")
    print("\n--- 計算結果摘要 (MAE %) ---")
    for key, value in output_stats.items():
        # 僅顯示 MAE %
        if key.endswith("MAE_%"):
            print(f"{key}: {value:.2f}%")
except Exception as e:
    print(f"計算結果時發生錯誤: {e}")
