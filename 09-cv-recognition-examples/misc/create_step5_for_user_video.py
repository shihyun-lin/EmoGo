"""
复制 Step 4 创建 Step 5，用于测试用户自己的影片
"""

import json
import sys

# 读取 notebook
notebook_path = sys.argv[1] if len(sys.argv) > 1 else '/Users/svjjsjrjs/Documents/心理學與神經資訊/Info_13_examples/R13546008_HW14.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 找到 Step 4 的所有 cells
step4_cells = []
step4_started = False

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        
        # 检测 Step 4 开始
        if '## Step 4' in source:
            step4_started = True
        
        # 检测其他 Step 开始（Step 4 结束）
        elif step4_started and ('## Step 5' in source or '## 結論' in source or '## Conclusion' in source):
            break
    
    # 收集 Step 4 的 cells
    if step4_started:
        step4_cells.append(cell)

print(f"✅ 找到 Step 4 的 {len(step4_cells)} 个 cells")

# 创建 Step 5 的 cells（复制并修改）
step5_cells = []

for i, cell in enumerate(step4_cells):
    new_cell = cell.copy()
    
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        
        # 修改标题
        if i == 0 and '## Step 4' in source:
            new_cell['source'] = [
                "## Step 5: 使用訓練好的模型測試您自己的影片\n",
                "\n",
                "在這個章節中，您可以測試自己的影片，調整不同的參數設定。\n",
                "\n",
                "**與 Step 4 的差異**：\n",
                "- Step 4 分析 `vlog.mp4`（示範影片）\n",
                "- Step 5 分析您自己的影片 ← 在這裡修改影片路徑和參數"
            ]
        else:
            # 将所有 "4." 替换为 "5."
            new_source = []
            for line in cell['source']:
                new_source.append(line.replace('### 4.', '### 5.'))
            new_cell['source'] = new_source
    
    elif cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # 修改影片路径和输出路径
        new_source = []
        for line in cell['source']:
            # 修改影片路径（添加注释提示用户修改）
            if "video_path = 'vlog.mp4'" in line or 'video_path =' in line:
                new_source.append("# ⚠️ 請修改為您自己的影片路徑\n")
                new_source.append("video_path = 'your_video.mp4'  # ← 改成您的影片檔名\n")
            # 修改输出路径
            elif "output_path = 'vlog_lr_emotion_results.csv'" in line or 'output_path =' in line:
                new_source.append("output_path = 'your_video_emotion_results.csv'\n")
            # 修改 sample_interval（添加注释）
            elif 'sample_interval = fps' in line:
                new_source.append("# ⚠️ 調整採樣率（越小越密集，處理時間越長）\n")
                new_source.append("# 選項: fps (每秒1幀), fps//2 (每秒2幀), fps//3 (每秒3幀), fps//5 (每秒5幀)\n")
                new_source.append("sample_interval = fps // 3  # ← 調整這裡\n")
            else:
                new_source.append(line)
        
        new_cell['source'] = new_source
    
    step5_cells.append(new_cell)

print(f"✅ 创建了 Step 5 的 {len(step5_cells)} 个 cells")

# 在 notebook 末尾添加 Step 5
for cell in step5_cells:
    nb['cells'].append(cell)

# 保存
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print(f"\n✅ 已成功添加 Step 5 到 notebook")
print(f"\n新增的 Step 5 包括:")
print("  - 5.1: 定義影片幀特徵萃取函數")
print("  - 5.2: 使用訓練好的模型進行逐幀辨識")
print("  - 5.3: 視覺化模型比較")
print("  - 5.4: 時間序列視覺化")
print("  - 5.5: 結論與觀察")
print("\n⚠️  記得修改:")
print("  1. video_path = 'your_video.mp4'  ← 改成您的影片路徑")
print("  2. sample_interval = fps // 3     ← 調整採樣率")
