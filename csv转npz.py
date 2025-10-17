import numpy as np
import pandas as pd

# 读取 CSV 文件（无表头）
csv_path = '2022-2025_filled.csv'
data = pd.read_csv(csv_path, header=None).values  # shape: (17856, 170)

print(data.shape)
# 扩展维度 -> (17856, 170, 1)
data_expanded = np.expand_dims(data, axis=-1)

# 保存为 .npz 文件
np.savez('PM25.npz', data=data_expanded)

# 打印确认信息
print("保存成功，数据 shape:", data_expanded.shape)
