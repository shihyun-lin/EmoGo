"""
✅ 修正 FER 影像載入問題的解決方案
問題原因:
1. 使用 Google Colab 路徑而非本地路徑
2. Excel 檔案中的副檔名是 .tif，但實際檔案是 .jpg
3. FER 需要正確的完整檔案路徑

使用方式: 在 Jupyter Notebook 中執行此程式碼
"""

import os
import cv2
import pandas as pd
from pathlib import Path
from fer import FER

# ========================================
# 1. 設定正確的本地路徑
# ========================================
# 根據你的系統調整基礎路徑
BASE_DIR = Path('/Users/svjjsjrjs/Documents/心理學與神經資訊/Info_13_examples')
IMAGE_FOLDER = BASE_DIR / 'Taiwanese' / 'faces_256x256'
EXCEL_PATH = BASE_DIR / 'Taiwanese' / 'Image_info.xls'

print(f"📂 影像資料夾: {IMAGE_FOLDER}")
print(f"📂 Excel 路徑: {EXCEL_PATH}")
print(f"✅ 路徑存在: {IMAGE_FOLDER.exists()}")

# ========================================
# 2. 建立檔名映射函數（處理 .tif → .jpg 轉換）
# ========================================
def get_actual_image_path(filename):
    """
    將 Excel 中的 .tif 檔名轉換為實際的 .jpg 路徑
    
    Args:
        filename: Excel 中的檔名（例如: 0221c08.tif 或 0221c08）
    
    Returns:
        實際的完整檔案路徑，如果檔案不存在則返回 None
    """
    if pd.isna(filename):
        return None
    
    # 去除原有的副檔名
    base_name = Path(filename).stem
    
    # 嘗試 .jpg 和 .tif 兩種副檔名
    for ext in ['.jpg', '.tif', '.jpeg', '.png']:
        img_path = IMAGE_FOLDER / f"{base_name}{ext}"
        if img_path.exists():
            return str(img_path)
    
    print(f"⚠️  找不到檔案: {base_name}")
    return None

# ========================================
# 3. 測試單一影像載入（使用 FER）
# ========================================
def test_single_image_fer(filename='0101a02.tif'):
    """
    測試單一影像的 FER 情緒辨識
    """
    print(f"\n🧪 測試 FER 載入: {filename}")
    print("="*50)
    
    # 取得實際路徑
    img_path = get_actual_image_path(filename)
    if img_path is None:
        print(f"❌ 檔案不存在: {filename}")
        return None
    
    print(f"✅ 實際路徑: {img_path}")
    
    # 載入影像
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ cv2.imread 載入失敗!")
        return None
    
    print(f"✅ 影像載入成功! 大小: {img.shape}")
    
    # 初始化 FER 偵測器
    emotion_detector = FER(mtcnn=True)
    
    # 偵測情緒
    result = emotion_detector.detect_emotions(img)
    print(f"✅ FER 偵測成功!")
    print(f"結果: {result}")
    
    return result

# ========================================
# 4. 批次處理函數
# ========================================
def process_taiwanese_dataset():
    """
    處理整個台灣臉孔資料集
    """
    # 載入 Excel
    df = pd.read_excel(EXCEL_PATH)
    df = df[df['file_name'].notna()]  # 移除空值
    
    print(f"\n📊 資料集大小: {len(df)} 筆")
    
    # 初始化 FER
    emotion_detector = FER(mtcnn=True)
    
    # 儲存結果
    results = []
    
    for idx, row in df.iterrows():
        filename = row['file_name']
        
        # 取得實際路徑
        img_path = get_actual_image_path(filename)
        if img_path is None:
            continue
        
        # 載入影像
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️  載入失敗: {filename}")
            continue
        
        # 偵測情緒
        try:
            emotion_result = emotion_detector.detect_emotions(img)
            if emotion_result and len(emotion_result) > 0:
                # 取得最高分數的情緒
                emotions = emotion_result[0]['emotions']
                predicted_emotion = max(emotions, key=emotions.get)
                
                results.append({
                    'filename': filename,
                    'ground_truth': row.get('maxIntCategory', 'unknown'),
                    'predicted': predicted_emotion,
                    'confidence': emotions[predicted_emotion],
                    'all_scores': emotions
                })
        except Exception as e:
            print(f"⚠️  辨識失敗 {filename}: {e}")
    
    print(f"\n✅ 成功處理 {len(results)} / {len(df)} 張影像")
    
    # 轉換為 DataFrame
    results_df = pd.DataFrame(results)
    return results_df

# ========================================
# 5. 執行測試
# ========================================
if __name__ == "__main__":
    # 測試單一影像
    print("### 步驟 1: 測試單一影像 ###")
    test_result = test_single_image_fer('0101a02.tif')
    
    # 測試檔名映射
    print("\n### 步驟 2: 測試檔名映射 ###")
    test_files = ['0101a02.tif', '0221c08.tif', '0221c08']
    for filename in test_files:
        path = get_actual_image_path(filename)
        print(f"{filename:20s} -> {path}")
