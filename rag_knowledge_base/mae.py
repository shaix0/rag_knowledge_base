"""Script to compute statistics on nutrition predictions using string inputs.

�o�Ӹ}������ Ground Truth �M Prediction �� JSON ���e�r��A�p�⵴�省���~�t (MAE) 
�M�ʤ��񥭧��~�t (MAE %)�A�o�ǫ��лP Nutrition5k �פ夤�Ω�����ҫ������Ьۦ��C
"""

import json
import statistics
import sys
import io 

DISH_ID_INDEX = 0
# �w�q���W�٪����ǡA�o�����P��J�� JSON �}�C�����ƾڶ��Ǥ@�P�C
DATA_FIELDNAMES = ["dish_id", "calories", "mass", "fat", "carb", "protein"]


def ParseJsonData(json_content: str) -> dict:
  """�ѪR JSON �榡���r��ƾڡA�åH dish_id (�Ĥ@��) ����إߦr��C
  
  �w�� JSON �榡��:
  [ ["dish_id", "calories", "mass", "fat", "carb", "protein"], ...]
  """
  parsed_data = {}
  try:
    # �ץ� 1: �����r��e�᪺�ťթM����šA�H�T�O JSON �ѪR�����T��
    cleaned_content = json_content.strip()

    # �ץ� 2: �B�z�D�з� JSON �榡 (�Ҧp�ϥγ�޸� ' �N���зǪ����޸� ")
    # �`�N: �o�Ӵ����O���F����`�����~�A���з� JSON ���ϥ����޸��C
    standardized_content = cleaned_content.replace("'", '"')
    
    # ���ոѪR JSON �r��A�w�����h�O�}�C
    data_list_of_lists = json.loads(standardized_content)
    
    if not isinstance(data_list_of_lists, list):
        print("JSON �ѪR���~: �w�����h���c���C�� (Array)�C")
        return {}

    expected_len = len(DATA_FIELDNAMES)
    
    for item_list in data_list_of_lists:
        # ���ҨC�Ӥ����C�����c
        if not isinstance(item_list, list) or len(item_list) != expected_len:
             print(f"ĵ�i: JSON ���إ������]�t {expected_len} �Ӥ������C��A�w���L: {item_list}")
             continue
        
        # ���o���a ID �@���r�媺��
        dish_id = str(item_list[DISH_ID_INDEX]).strip()
        if not dish_id: continue
        
        # �N���ǦC��ƾ��ഫ���a�����W�٪��r��
        item_dict = {}
        for i, field_name in enumerate(DATA_FIELDNAMES):
            # �ƾڥH�r��Φ��x�s�A�d�ݭp����ഫ���B�I��
            item_dict[field_name] = str(item_list[i])
            
        parsed_data[dish_id] = item_dict 
    
  except json.JSONDecodeError as e:
    print(f"JSON �ѪR���~: ���ˬd��J�r��O�_�����Ī� JSON �榡: {e}")
    return {}
    
  return parsed_data

# --- �d�ҿ�J�ƾ� (�бN�z���u�� JSON ���e�K�J�o��) ---

# �d�� Ground Truth �ƾ� (�o�O�u���)
# �榡: [ [dish_id, calories, mass, fat, carb, protein], ... ]
GROUNDTRUTH_JSON_CONTENT = """
[
  ['dish_20251002', '583.40', '481.00', '23.30', '52.40', '42.30']
]
"""

# �d�� Prediction �ƾ� (�o�O�ҫ��w����)
PREDICTIONS_JSON_CONTENT = """
[
  ['dish_20251002', '622.30', '530.00', '17.25', '77.55', '38.97'],
  ['dish_20251002', '371.80', '410.00', '2.13', '77.75', '12.40'],
  ['dish_20251002', '371.80', '410.00', '2.13', '77.75', '12.40'],
  ['dish_20251002', '381.30', '440.00', '2.24', '79.67', '13.02'],
  ['dish_20251002', '498.30', '460.00', '10.34', '61.47', '40.04']
]
"""

# ��X�� JSON �ɮ׸��| (���M�ݭn�@�Ӹ��|���x�s�έp���G)
output_path = "output_statistics.json" 

# -----------------------------------------------------

# 1. �ѪR��J�r��
print("�ѪR Ground Truth JSON �r��...")
groundtruth_data = ParseJsonData(GROUNDTRUTH_JSON_CONTENT)
print("�ѪR Predictions JSON �r��...")
prediction_data = ParseJsonData(PREDICTIONS_JSON_CONTENT)

# 2. �ƾڳB�z�P�έp�p��
groundtruth_values = {}
err_values = {}
output_stats = {}

for field in DATA_FIELDNAMES[1:]:
  groundtruth_values[field] = []
  err_values[field] = []

# �ȳB�z�P�ɦs�b�� Ground Truth �M Prediction ���� dish_id
common_dish_ids = set(groundtruth_data.keys()) & set(prediction_data.keys())
    
print(f"��� {len(common_dish_ids)} �Ӧ@�P�����a ID �i�����C")


for dish_id in common_dish_ids:
  # �q�ƾڤ������ƭ� (�{�b�O�r��A�ϥ� key �s��)
  gt_dict = groundtruth_data[dish_id]
  pred_dict = prediction_data[dish_id]
  
  # �M����i����� (calories, mass, fat, carb, protein)
  for field_name in DATA_FIELDNAMES[1:]:
    try:
        # �ϥ� .get() �w���a�q�r�夤�����ƭ�
        # �ѩ�ƾڬO�H�r��Φ��x�s���A�o�̱N�����ഫ���B�I��
        gt_value = float(gt_dict.get(field_name, 0.0))
        pred_value = float(pred_dict.get(field_name, 0.0))
        
        groundtruth_values[field_name].append(gt_value)
        err_values[field_name].append(abs(pred_value - gt_value))
    except (TypeError, ValueError):
        print(f"ĵ�i: ���a {dish_id}, ��� {field_name} ���ƭ��ഫ���ѡA���L�Ӽƾ��I�C")
        continue

# 3. �p�� MAE �M MAE %
for field in DATA_FIELDNAMES[1:]:
    if groundtruth_values[field]:
        mean_err = statistics.mean(err_values[field])
        mean_gt = statistics.mean(groundtruth_values[field])
        
        output_stats[field + "_MAE"] = mean_err
        
        # �קK���H�s
        if mean_gt != 0:
             output_stats[field + "_MAE_%"] = 100 * mean_err / mean_gt
        else:
             output_stats[field + "_MAE_%"] = 0.0
    else:
        output_stats[field + "_MAE"] = 0.0
        output_stats[field + "_MAE_%"] = 0.0

# 4. �g�J���G
try:
    # �������ɮ׼g�J�A�����b����x��X���G
    # with open(output_path, "w") as f_out:
    #   f_out.write(json.dumps(output_stats, indent=2))
      
    print(f"\n�έp�p�⧹���C")
    print("\n--- �p�⵲�G�K�n (MAE) ---")
    for key, value in output_stats.items():
        # ����� MAE
        if key.endswith("MAE"):
            print(f"{key}: {value:.2f}")
    print("\n--- �p�⵲�G�K�n (MAE %) ---")
    for key, value in output_stats.items():
        # ����� MAE %
        if key.endswith("MAE_%"):
            print(f"{key}: {value:.2f}%")
except Exception as e:
    print(f"�p�⵲�G�ɵo�Ϳ��~: {e}")
