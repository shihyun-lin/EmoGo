"""
修正 R13546008_HW14.ipynb 中 Step 3.5.1 的訓練集/驗證集切分邏輯
避免數據洩漏：確保訓練集不包含驗證集的 106 張圖片
"""

import json
import sys

# 讀取 notebook
notebook_path = sys.argv[1] if len(sys.argv) > 1 else '/Users/svjjsjrjs/Documents/心理學與神經資訊/Info_13_examples/R13546008_HW14.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 新的正確代碼
new_code = [
    "# =========================\n",
    "# 正確的訓練集/驗證集切分\n",
    "# =========================\n",
    "\n",
    "# 驗證集：106 張乾淨樣本（之前篩選過的）\n",
    "val_df = clean_df.copy()  # clean_df 是經過 EntropyVal, maxInt, FACS 篩選的 106 張\n",
    "\n",
    "# 訓練集：從 2272 張中排除驗證集的 106 張\n",
    "# 使用 index 來排除驗證集樣本，避免數據洩漏\n",
    "train_df = df[~df.index.isin(val_df.index)].copy()\n",
    "\n",
    "print(f\"原始完整資料集: {len(df)} 張\")\n",
    "print(f\"訓練集大小: {len(train_df)} 張 (應為 2166)\")\n",
    "print(f\"驗證集大小: {len(val_df)} 張 (應為 106)\")\n",
    "print(f\"總計: {len(train_df) + len(val_df)} 張 (應等於 {len(df)})\")\n",
    "\n",
    "print(f\"\\n訓練集情緒分布:\")\n",
    "print(train_df['emotion_label'].value_counts())\n",
    "print(f\"\\n驗證集情緒分布:\")\n",
    "print(val_df['emotion_label'].value_counts())\n",
    "\n",
    "# 驗證沒有重疊\n",
    "overlap = set(train_df.index) & set(val_df.index)\n",
    "print(f\"\\n✅ 訓練集與驗證集重疊樣本數: {len(overlap)} (應為 0)\")\n",
    "if len(overlap) > 0:\n",
    "    print(f\"⚠️  警告：發現重疊樣本！\")\n",
    "else:\n",
    "    print(f\"✅ 確認：訓練集與驗證集完全分離，無數據洩漏\")"
]

# 找到 Step 3.5.1 的 code cell
found = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        # 尋找包含 "建立訓練集：所有 2272 張圖" 的 cell
        if '建立訓練集：所有 2272 張圖' in source or ('train_df = df.copy()' in source and 'val_df = clean_df.copy()' in source):
            print(f"✅ 找到需要修正的 cell (index {i})")
            # 替換為新的正確代碼
            nb['cells'][i]['source'] = new_code
            found = True
            break

if not found:
    print("❌ 找不到需要修正的 cell")
    sys.exit(1)

# 儲存更新後的 notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print(f"✅ 已成功更新 notebook: {notebook_path}")
print("\n修正內容：")
print("  - 訓練集改為從 2272 張中排除驗證集的 106 張")
print("  - 訓練集大小：2166 張 (2272 - 106)")
print("  - 驗證集大小：106 張")
print("  - 新增重疊檢查，確保無數據洩漏")
