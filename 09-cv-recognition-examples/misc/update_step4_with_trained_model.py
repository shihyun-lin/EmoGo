"""
更新 R13546008_HW14.ipynb 中的 Step 4
使用訓練好的 Logistic Regression 模型進行影片情緒辨識
"""

import json
import sys

# 讀取 notebook
notebook_path = sys.argv[1] if len(sys.argv) > 1 else '/Users/svjjsjrjs/Documents/心理學與神經資訊/Info_13_examples/R13546008_HW14.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 新的 Step 4 cells（使用訓練好的模型）
new_step4_cells = [
    # Markdown: 章節標題
    {
        "cell_type": "markdown",
        "metadata": {"id": "G7iEUyBQdo7u"},
        "source": [
            "## Step 4: 使用訓練好的模型進行影片情緒辨識（vlog.mp4）\n",
            "\n",
            "在這個章節中，我們將使用 **Step 3.5 訓練好的 Logistic Regression 模型**對影片進行逐幀情緒辨識。\n",
            "\n",
            "**流程**：\n",
            "1. 載入影片並逐幀提取畫面\n",
            "2. 對每一幀提取 DeepFace embeddings (512-dim) + FER features (7-dim)\n",
            "3. 使用訓練好的 `scaler` 和 `lr_model` 進行預測\n",
            "4. 視覺化結果並與 DeepFace zero-shot 比較"
        ]
    },
    
    # Markdown: 4.1
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4.1 定義影片幀特徵萃取函數"
        ]
    },
    
    # Code: 定義特徵萃取函數
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def extract_frame_features(frame):\n",
            "    \"\"\"\n",
            "    從單一影片幀中萃取 519 維特徵向量\n",
            "    \n",
            "    Args:\n",
            "        frame: OpenCV 讀取的影像 (numpy array)\n",
            "    \n",
            "    Returns:\n",
            "        features (np.array): 519 維特徵向量 [DeepFace(512) + FER(7)]\n",
            "        失敗則返回 None\n",
            "    \"\"\"\n",
            "    try:\n",
            "        # 1. 萃取 DeepFace embedding (512 維)\n",
            "        deepface_result = DeepFace.represent(\n",
            "            img_path=frame,\n",
            "            model_name='VGG-Face',\n",
            "            enforce_detection=False,\n",
            "            detector_backend='opencv'\n",
            "        )\n",
            "        \n",
            "        if isinstance(deepface_result, list) and len(deepface_result) > 0:\n",
            "            deepface_embedding = np.array(deepface_result[0]['embedding'])\n",
            "        else:\n",
            "            return None\n",
            "        \n",
            "        # 2. 萃取 FER 特徵 (7 維)\n",
            "        fer_result = fer_detector.detect_emotions(frame)\n",
            "        \n",
            "        if fer_result is None or len(fer_result) == 0:\n",
            "            return None\n",
            "        \n",
            "        emotions = fer_result[0]['emotions']\n",
            "        fer_features = np.array([\n",
            "            emotions['angry'],\n",
            "            emotions['disgust'],\n",
            "            emotions['fear'],\n",
            "            emotions['happy'],\n",
            "            emotions['sad'],\n",
            "            emotions['surprise'],\n",
            "            emotions['neutral']\n",
            "        ])\n",
            "        \n",
            "        # 3. 合併特徵\n",
            "        combined_features = np.concatenate([deepface_embedding, fer_features])\n",
            "        \n",
            "        return combined_features\n",
            "        \n",
            "    except Exception as e:\n",
            "        return None\n",
            "\n",
            "print(\"✅ 影片幀特徵萃取函數定義完成\")"
        ]
    },
    
    # Markdown: 4.2
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4.2 使用訓練好的模型進行影片逐幀辨識"
        ]
    },
    
    # Code: 影片處理
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "e2axR08Odo7u"},
        "outputs": [],
        "source": [
            "import cv2\n",
            "from tqdm import tqdm\n",
            "\n",
            "# ✅ 修正：使用本地路徑\n",
            "video_path = 'vlog.mp4'  # 本地路徑\n",
            "output_path = 'vlog_lr_emotion_results.csv'\n",
            "\n",
            "# 開啟影片\n",
            "cap = cv2.VideoCapture(video_path)\n",
            "fps = int(cap.get(cv2.CAP_PROP_FPS))\n",
            "total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n",
            "\n",
            "print(f\"影片資訊:\")\n",
            "print(f\"  FPS: {fps}\")\n",
            "print(f\"  總幀數: {total_frames}\")\n",
            "print(f\"  影片長度: {total_frames / fps:.1f} 秒\")\n",
            "\n",
            "# 每秒抽 1 幀進行分析\n",
            "sample_interval = fps\n",
            "frame_count = 0\n",
            "results_lr = []  # Logistic Regression 結果\n",
            "results_deepface = []  # DeepFace zero-shot 結果（用於比較）\n",
            "failed_frames = 0\n",
            "\n",
            "print(f\"\\n開始處理影片（每秒抽取 1 幀）...\")\n",
            "\n",
            "# 計算預期處理的幀數\n",
            "expected_samples = total_frames // sample_interval\n",
            "\n",
            "with tqdm(total=expected_samples, desc=\"處理影片幀\") as pbar:\n",
            "    while cap.isOpened():\n",
            "        ret, frame = cap.read()\n",
            "        if not ret:\n",
            "            break\n",
            "        \n",
            "        # 只處理每秒的第一幀\n",
            "        if frame_count % sample_interval == 0:\n",
            "            time_sec = frame_count // fps\n",
            "            \n",
            "            # 方法 1: 使用訓練好的 Logistic Regression\n",
            "            features = extract_frame_features(frame)\n",
            "            \n",
            "            if features is not None:\n",
            "                # 標準化特徵\n",
            "                features_scaled = scaler.transform(features.reshape(1, -1))\n",
            "                \n",
            "                # 使用 Logistic Regression 預測\n",
            "                emotion_lr = lr_model.predict(features_scaled)[0]\n",
            "                \n",
            "                results_lr.append({\n",
            "                    'time_sec': time_sec,\n",
            "                    'emotion': emotion_lr\n",
            "                })\n",
            "            else:\n",
            "                failed_frames += 1\n",
            "                results_lr.append({\n",
            "                    'time_sec': time_sec,\n",
            "                    'emotion': 'unknown'\n",
            "                })\n",
            "            \n",
            "            # 方法 2: DeepFace zero-shot（用於比較）\n",
            "            try:\n",
            "                result = DeepFace.analyze(\n",
            "                    frame,\n",
            "                    actions=['emotion'],\n",
            "                    detector_backend='opencv',\n",
            "                    enforce_detection=False,\n",
            "                    silent=True\n",
            "                )\n",
            "                result = result[0] if isinstance(result, list) else result\n",
            "                emotion_deepface = result['dominant_emotion']\n",
            "            except:\n",
            "                emotion_deepface = 'unknown'\n",
            "            \n",
            "            results_deepface.append({\n",
            "                'time_sec': time_sec,\n",
            "                'emotion': emotion_deepface\n",
            "            })\n",
            "            \n",
            "            pbar.update(1)\n",
            "        \n",
            "        frame_count += 1\n",
            "\n",
            "cap.release()\n",
            "\n",
            "# 轉換為 DataFrame\n",
            "results_lr_df = pd.DataFrame(results_lr)\n",
            "results_deepface_df = pd.DataFrame(results_deepface)\n",
            "\n",
            "# 儲存結果\n",
            "results_lr_df.to_csv(output_path, index=False)\n",
            "\n",
            "print(f\"\\n✅ 影片處理完成\")\n",
            "print(f\"   分析時間點數: {len(results_lr)}\")\n",
            "print(f\"   失敗幀數: {failed_frames}\")\n",
            "print(f\"   結果已儲存至: {output_path}\")\n",
            "\n",
            "print(\"\\nLogistic Regression 結果預覽:\")\n",
            "print(results_lr_df.head(10))"
        ]
    },
    
    # Markdown: 4.3
    {
        "cell_type": "markdown",
        "metadata": {"id": "PKvlKRJudo7v"},
        "source": [
            "### 4.3 視覺化：模型比較（Logistic Regression vs DeepFace）"
        ]
    },
    
    # Code: 視覺化比較
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "D5GTZfhXdo7v"},
        "outputs": [],
        "source": [
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "# 統計情緒分布\n",
            "lr_emotion_counts = results_lr_df['emotion'].value_counts()\n",
            "deepface_emotion_counts = results_deepface_df['emotion'].value_counts()\n",
            "\n",
            "# 並排比較\n",
            "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
            "\n",
            "# 左圖：Logistic Regression\n",
            "axes[0].bar(lr_emotion_counts.index, lr_emotion_counts.values, color='#27ae60', alpha=0.8)\n",
            "axes[0].set_title('Logistic Regression (Trained Model)', fontsize=14, fontweight='bold')\n",
            "axes[0].set_xlabel('Emotion')\n",
            "axes[0].set_ylabel('Frequency (seconds)')\n",
            "axes[0].tick_params(axis='x', rotation=45)\n",
            "axes[0].grid(axis='y', alpha=0.3)\n",
            "\n",
            "# 右圖：DeepFace zero-shot\n",
            "axes[1].bar(deepface_emotion_counts.index, deepface_emotion_counts.values, color='#3498db', alpha=0.8)\n",
            "axes[1].set_title('DeepFace (Zero-shot)', fontsize=14, fontweight='bold')\n",
            "axes[1].set_xlabel('Emotion')\n",
            "axes[1].set_ylabel('Frequency (seconds)')\n",
            "axes[1].tick_params(axis='x', rotation=45)\n",
            "axes[1].grid(axis='y', alpha=0.3)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "# 詳細情緒佔比\n",
            "print(\"=\"*60)\n",
            "print(\"Logistic Regression 情緒分布:\")\n",
            "print(\"=\"*60)\n",
            "for emotion, count in lr_emotion_counts.items():\n",
            "    print(f\"  {emotion:12s}: {count:3d} 秒 ({count/len(results_lr_df):.1%})\")\n",
            "\n",
            "print(\"\\n\" + \"=\"*60)\n",
            "print(\"DeepFace (Zero-shot) 情緒分布:\")\n",
            "print(\"=\"*60)\n",
            "for emotion, count in deepface_emotion_counts.items():\n",
            "    print(f\"  {emotion:12s}: {count:3d} 秒 ({count/len(results_deepface_df):.1%})\")"
        ]
    },
    
    # Markdown: 4.4
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4.4 時間序列視覺化：情緒隨時間變化"
        ]
    },
    
    # Code: 時間序列圖
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 為情緒分配顏色\n",
            "emotion_colors = {\n",
            "    'happy': '#f1c40f',\n",
            "    'sad': '#3498db',\n",
            "    'angry': '#e74c3c',\n",
            "    'disgust': '#9b59b6',\n",
            "    'fear': '#95a5a6',\n",
            "    'surprise': '#e67e22',\n",
            "    'neutral': '#34495e',\n",
            "    'unknown': '#ecf0f1'\n",
            "}\n",
            "\n",
            "# 繪製時間序列\n",
            "fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)\n",
            "\n",
            "# 上圖：Logistic Regression\n",
            "for i, row in results_lr_df.iterrows():\n",
            "    color = emotion_colors.get(row['emotion'], '#000000')\n",
            "    axes[0].axvspan(row['time_sec'], row['time_sec'] + 1, \n",
            "                    color=color, alpha=0.7)\n",
            "\n",
            "axes[0].set_ylabel('Emotion', fontsize=12)\n",
            "axes[0].set_title('Logistic Regression: Emotion Timeline', fontsize=14, fontweight='bold')\n",
            "axes[0].set_ylim([0, 1])\n",
            "axes[0].set_yticks([])\n",
            "axes[0].grid(axis='x', alpha=0.3)\n",
            "\n",
            "# 下圖：DeepFace\n",
            "for i, row in results_deepface_df.iterrows():\n",
            "    color = emotion_colors.get(row['emotion'], '#000000')\n",
            "    axes[1].axvspan(row['time_sec'], row['time_sec'] + 1, \n",
            "                    color=color, alpha=0.7)\n",
            "\n",
            "axes[1].set_xlabel('Time (seconds)', fontsize=12)\n",
            "axes[1].set_ylabel('Emotion', fontsize=12)\n",
            "axes[1].set_title('DeepFace (Zero-shot): Emotion Timeline', fontsize=14, fontweight='bold')\n",
            "axes[1].set_ylim([0, 1])\n",
            "axes[1].set_yticks([])\n",
            "axes[1].grid(axis='x', alpha=0.3)\n",
            "\n",
            "# 添加圖例\n",
            "from matplotlib.patches import Patch\n",
            "legend_elements = [Patch(facecolor=color, label=emotion, alpha=0.7) \n",
            "                   for emotion, color in emotion_colors.items()]\n",
            "fig.legend(handles=legend_elements, loc='upper right', ncol=4, \n",
            "           bbox_to_anchor=(0.98, 0.98))\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    
    # Markdown: 結論
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4.5 結論與觀察\n",
            "\n",
            "**方法比較**：\n",
            "- **Logistic Regression (訓練模型)**：使用在 2166 張台灣華人臉孔上訓練的模型，理論上對台灣人臉的情緒特徵更敏感\n",
            "- **DeepFace (Zero-shot)**：使用預訓練模型直接進行推論，可能對西方面孔表現較好\n",
            "\n",
            "**觀察重點**：\n",
            "1. 兩個模型在影片中識別到的主要情緒是否一致？\n",
            "2. Logistic Regression 是否能更穩定地識別特定情緒（如 happy、sad）？\n",
            "3. 哪些時間段兩個模型的預測結果差異較大？\n",
            "\n",
            "**改進方向**：\n",
            "- 可以考慮使用滑動窗口平滑結果（減少幀間跳躍）\n",
            "- 增加採樣率（例如每秒 2-3 幀）以捕捉更細緻的情緒變化\n",
            "- 結合多個模型的預測結果（ensemble）以提高穩定性"
        ]
    }
]

# 找到 Step 4 開始的位置
step4_start_index = None
step4_end_index = None

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        
        # 找到 Step 4 開始
        if step4_start_index is None and '## Step 4' in source:
            step4_start_index = i
        
        # 找到 Step 5 開始（如果有的話）
        if step4_start_index is not None and step4_end_index is None:
            if '## Step 5' in source or '## 結論' in source or '## Conclusion' in source:
                step4_end_index = i
                break

# 如果沒有 Step 5，則刪除到文件末尾
if step4_start_index is not None and step4_end_index is None:
    step4_end_index = len(nb['cells'])

if step4_start_index is None:
    print("❌ 找不到 Step 4")
    sys.exit(1)

print(f"✅ 找到 Step 4 位置: index {step4_start_index} 到 {step4_end_index}")
print(f"   將刪除 {step4_end_index - step4_start_index} 個舊的 cells")
print(f"   將插入 {len(new_step4_cells)} 個新的 cells")

# 刪除舊的 Step 4 cells
for _ in range(step4_end_index - step4_start_index):
    nb['cells'].pop(step4_start_index)

# 插入新的 Step 4 cells
for i, new_cell in enumerate(new_step4_cells):
    nb['cells'].insert(step4_start_index + i, new_cell)

# 儲存更新後的 notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print(f"\n✅ 已成功更新 notebook: {notebook_path}")
print(f"\n新的 Step 4 包括:")
print("  - 4.1: 定義影片幀特徵萃取函數")
print("  - 4.2: 使用訓練好的模型進行逐幀辨識")
print("  - 4.3: 視覺化模型比較（LR vs DeepFace）")
print("  - 4.4: 時間序列視覺化")
print("  - 4.5: 結論與觀察")
