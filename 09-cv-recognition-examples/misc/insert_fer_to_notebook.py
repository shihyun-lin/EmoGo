"""
將 FER 實作插入到 R13546008_HW14.ipynb 的腳本
這個腳本會在現有的 FER 測試程式碼之後，Step 4 之前，插入完整的 FER 實作章節
"""

import json
import sys

# 讀取 notebook
notebook_path = sys.argv[1] if len(sys.argv) > 1 else '/Users/svjjsjrjs/Documents/心理學與神經資訊/Info_13_examples/R13546008_HW14.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 新的 FER 實作章節
new_cells = [
    # Markdown: 章節標題
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 2.5: 使用 FER 模型進行驗證（替代方案）\n",
            "\n",
            "在這個章節中，我們使用 **FER (Facial Emotion Recognition)** 函式庫作為替代模型，與 DeepFace 進行比較。FER 使用 MTCNN 進行臉部偵測，可能對華人臉孔有不同的表現。"
        ]
    },
    
    # Code: 修正路徑並定義 FER 辨識函數
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from fer import FER\n",
            "import cv2\n",
            "\n",
            "# ✅ 修正：使用本地路徑（與 DeepFace 一致）\n",
            "# 注意：這裡改用本地相對路徑，而非 Colab 路徑\n",
            "image_folder_local = 'Taiwanese/faces_256x256'\n",
            "\n",
            "# 初始化 FER 偵測器（使用 MTCNN）\n",
            "fer_detector = FER(mtcnn=True)\n",
            "\n",
            "def recognize_emotion_fer(image_path):\n",
            "    \"\"\"\n",
            "    使用 FER 對單張影像進行情緒辨識\n",
            "    \n",
            "    Args:\n",
            "        image_path: 完整的影像路徑\n",
            "    \n",
            "    Returns:\n",
            "        predicted_emotion (str): 預測的情緒類別，失敗則返回 None\n",
            "    \"\"\"\n",
            "    try:\n",
            "        # 載入影像\n",
            "        img = cv2.imread(image_path)\n",
            "        if img is None:\n",
            "            return None\n",
            "        \n",
            "        # 使用 top_emotion 取得最高分數的情緒\n",
            "        result = fer_detector.top_emotion(img)\n",
            "        \n",
            "        if result is None or result[0] is None:\n",
            "            return None\n",
            "        \n",
            "        # result 是 tuple: ('happy', 0.95)\n",
            "        return result[0]  # 返回情緒名稱\n",
            "        \n",
            "    except Exception as e:\n",
            "        # print(f\"辨識失敗: {e}\")\n",
            "        return None\n",
            "\n",
            "print(\"✅ FER 情緒辨識函式定義完成\")"
        ]
    },
    
    # Markdown: 測試單一影像
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2.5.1 測試 FER 單一影像載入"
        ]
    },
    
    # Code: 測試
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 測試單一影像（使用與 DeepFace 相同的 find_image_path 函數）\n",
            "test_filename = clean_df['file_name'].iloc[0]\n",
            "test_path = find_image_path(image_folder_local, test_filename)\n",
            "\n",
            "print(f\"測試檔名: {test_filename}\")\n",
            "print(f\"實際路徑: {test_path}\")\n",
            "\n",
            "if test_path:\n",
            "    img_test = cv2.imread(test_path)\n",
            "    print(f\"影像載入成功: {img_test is not None}\")\n",
            "    \n",
            "    if img_test is not None:\n",
            "        print(f\"影像大小: {img_test.shape}\")\n",
            "        \n",
            "        # 測試 FER 辨識\n",
            "        pred = recognize_emotion_fer(test_path)\n",
            "        print(f\"FER 預測情緒: {pred}\")\n",
            "else:\n",
            "    print(\"❌ 找不到測試影像\")"
        ]
    },
    
    # Markdown: 批次辨識
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2.5.2 對篩選樣本執行 FER 情緒辨識"
        ]
    },
    
    # Code: 批次處理
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from tqdm import tqdm\n",
            "\n",
            "# 批次辨識（邏輯與 DeepFace 相同）\n",
            "fer_predictions = []\n",
            "fer_ground_truths = []\n",
            "fer_used_index = []\n",
            "fer_missing_files = 0\n",
            "fer_failed_recognition = 0\n",
            "\n",
            "for idx, row in tqdm(clean_df.iterrows(), total=len(clean_df), desc=\"FER 辨識中\"):\n",
            "    # 使用相同的 find_image_path 函數\n",
            "    img_path = find_image_path(image_folder_local, row['file_name'])\n",
            "    \n",
            "    if img_path is None:\n",
            "        fer_missing_files += 1\n",
            "        continue\n",
            "    \n",
            "    # 執行 FER 辨識\n",
            "    pred = recognize_emotion_fer(img_path)\n",
            "    \n",
            "    if pred is not None:\n",
            "        fer_predictions.append(pred)\n",
            "        fer_ground_truths.append(row['emotion_label'])\n",
            "        fer_used_index.append(idx)\n",
            "    else:\n",
            "        fer_failed_recognition += 1\n",
            "\n",
            "print(f\"\\n✅ FER 成功辨識 {len(fer_predictions)} 張影像\")\n",
            "print(f\"❌ 找不到實體檔案的筆數: {fer_missing_files}\")\n",
            "print(f\"⚠️  辨識失敗的筆數: {fer_failed_recognition}\")\n",
            "\n",
            "# 建立結果 DataFrame\n",
            "fer_results_df = clean_df.loc[fer_used_index].copy()\n",
            "fer_results_df['gt_emotion'] = fer_ground_truths\n",
            "fer_results_df['predicted_emotion'] = fer_predictions\n",
            "\n",
            "print(\"\\nFER 結果前 5 筆：\")\n",
            "print(fer_results_df[['file_name', 'gt_emotion', 'predicted_emotion']].head())"
        ]
    },
    
    # Markdown: 計算準確率
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2.5.3 計算 FER 準確率指標"
        ]
    },
    
    # Code: 評估指標
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.metrics import classification_report, confusion_matrix, accuracy_score\n",
            "import seaborn as sns\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "# 計算整體準確率\n",
            "fer_accuracy = accuracy_score(fer_ground_truths, fer_predictions)\n",
            "print(f\"Overall accuracy (FER): {fer_accuracy:.3f}\")\n",
            "\n",
            "# 分類報告\n",
            "print(\"\\nClassification report (FER):\")\n",
            "print(classification_report(fer_ground_truths, fer_predictions))\n",
            "\n",
            "# Confusion Matrix\n",
            "fer_cm = confusion_matrix(fer_ground_truths, fer_predictions, \n",
            "                          labels=['happy', 'sad', 'angry','disgust', 'fear', 'surprise'])\n",
            "\n",
            "plt.figure(figsize=(8, 6))\n",
            "sns.heatmap(fer_cm, annot=True, fmt='d', cmap='Blues',\n",
            "            xticklabels=['happy', 'sad', 'angry', 'disgust', 'fear', 'surprise'],\n",
            "            yticklabels=['happy', 'sad', 'angry', 'disgust', 'fear', 'surprise'])\n",
            "plt.title('FER Confusion Matrix')\n",
            "plt.ylabel('Ground Truth')\n",
            "plt.xlabel('Predicted')\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print(\"\\n✅ FER 模型評估完成\")"
        ]
    },
    
    # Markdown: 比較 DeepFace vs FER
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2.5.4 DeepFace vs FER 模型比較\n",
            "\n",
            "在這裡我們比較兩個模型在台灣華人臉孔資料集上的表現差異。"
        ]
    },
    
    # Code: 模型比較
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 比較兩個模型的準確率\n",
            "print(\"=\" * 50)\n",
            "print(\"模型比較摘要\")\n",
            "print(\"=\" * 50)\n",
            "print(f\"DeepFace Overall Accuracy: {accuracy_score(ground_truths, predictions):.3f}\")\n",
            "print(f\"FER Overall Accuracy:      {fer_accuracy:.3f}\")\n",
            "print(\"\\n\")\n",
            "\n",
            "# 建立比較表\n",
            "comparison_df = pd.DataFrame({\n",
            "    'Model': ['DeepFace', 'FER'],\n",
            "    'Accuracy': [accuracy_score(ground_truths, predictions), fer_accuracy],\n",
            "    'Samples': [len(predictions), len(fer_predictions)]\n",
            "})\n",
            "\n",
            "print(comparison_df)\n",
            "print(\"\\n\")\n",
            "\n",
            "# 視覺化比較\n",
            "fig, ax = plt.subplots(figsize=(8, 5))\n",
            "ax.bar(comparison_df['Model'], comparison_df['Accuracy'], \n",
            "       color=['#3498db', '#e74c3c'], alpha=0.8)\n",
            "ax.set_ylabel('Accuracy')\n",
            "ax.set_title('DeepFace vs FER: Accuracy Comparison on Taiwanese Faces')\n",
            "ax.set_ylim([0, 1])\n",
            "\n",
            "# 在柱狀圖上顯示數值\n",
            "for i, v in enumerate(comparison_df['Accuracy']):\n",
            "    ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=12, fontweight='bold')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print(\"✅ 模型比較完成\")"
        ]
    }
]

# 找到 Step 4 的位置（在 "## Step 4" markdown cell 之前插入）
insert_index = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        if '## Step 4' in source or '##Step 4' in source:
            insert_index = i
            break

if insert_index is None:
    print("❌ 找不到 Step 4 的位置")
    sys.exit(1)

print(f"✅ 找到 Step 4 位置: index {insert_index}")
print(f"將在此之前插入 {len(new_cells)} 個新的 cell")

# 插入新的 cells
for i, new_cell in enumerate(new_cells):
    nb['cells'].insert(insert_index + i, new_cell)

# 儲存更新後的 notebook
output_path = notebook_path
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print(f"✅ 已成功更新 notebook: {output_path}")
print(f"新增了 {len(new_cells)} 個 cells")
print("\\n新增的章節包括:")
print("  - Step 2.5: 使用 FER 模型進行驗證")
print("  - 2.5.1: 測試單一影像")
print("  - 2.5.2: 批次辨識")
print("  - 2.5.3: 計算準確率")
print("  - 2.5.4: DeepFace vs FER 比較")
