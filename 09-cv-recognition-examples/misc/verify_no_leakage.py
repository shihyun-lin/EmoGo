"""
驗證訓練集/驗證集切分是否真的沒有數據洩漏
檢查 clean_df 的 index 與 df 的 index 是否一致
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("數據洩漏驗證測試")
print("=" * 60)

# 模擬原始數據集
np.random.seed(42)
df = pd.DataFrame({
    'id': range(100),  # 模擬 2272 張圖
    'value': np.random.rand(100)
})

print(f"\n1. 原始 df:")
print(f"   大小: {len(df)}")
print(f"   Index: {df.index.tolist()[:10]}... (showing first 10)")

# 模擬 clean_df 的創建（通過條件篩選）
clean_df = df[df['value'] > 0.8].copy()

print(f"\n2. clean_df (通過篩選創建):")
print(f"   大小: {len(clean_df)}")
print(f"   Index: {clean_df.index.tolist()}")
print(f"   ⚠️  注意：clean_df 的 index 保留了原始 df 的 index")

# 方法 1：使用 index 排除（我們的方法）
val_df = clean_df.copy()
train_df = df[~df.index.isin(val_df.index)].copy()

print(f"\n3. 使用 index 排除:")
print(f"   train_df 大小: {len(train_df)} (期望: {len(df) - len(clean_df)})")
print(f"   val_df 大小: {len(val_df)} (期望: {len(clean_df)})")

# 檢查重疊
overlap_indices = set(train_df.index) & set(val_df.index)
print(f"\n4. 重疊檢查:")
print(f"   重疊的 index: {overlap_indices}")
print(f"   重疊數量: {len(overlap_indices)}")

if len(overlap_indices) == 0:
    print(f"   ✅ 沒有重疊！訓練集和驗證集完全分離")
else:
    print(f"   ❌ 有重疊！存在數據洩漏")

# 驗證總數
total = len(train_df) + len(val_df)
print(f"\n5. 總數驗證:")
print(f"   train_df + val_df = {total}")
print(f"   原始 df = {len(df)}")
print(f"   是否相等: {total == len(df)}")

# 檢查是否有任何 df 中的樣本被遺漏
all_indices = set(train_df.index) | set(val_df.index)
original_indices = set(df.index)
missing = original_indices - all_indices
extra = all_indices - original_indices

print(f"\n6. 完整性檢查:")
print(f"   遺漏的樣本: {len(missing)}")
print(f"   多出的樣本: {len(extra)}")
if len(missing) == 0 and len(extra) == 0:
    print(f"   ✅ 所有樣本都被正確分配，沒有遺漏或重複")

print("\n" + "=" * 60)
print("結論:")
print("=" * 60)
print("使用 df[~df.index.isin(val_df.index)] 的方法是正確的！")
print("前提條件：clean_df 必須是從 df 通過條件篩選得到的")
print("因為 pandas 的 boolean indexing 會保留原始 index")
print("=" * 60)
