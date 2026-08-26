# data_preprocessing.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE


# 定义 MLP 用于特征压缩
class MLPFeatureCompression(nn.Module):
    def __init__(self, input_dim, compressed_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, compressed_dim)
        )

    def forward(self, x):
        return self.mlp(x)

# 数据加载函数
def load_arff(file_path):
    from scipy.io import arff
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    return df



# 处理类别不平衡（SMOTE）
def balance_data(X, y):
    # # 自动识别少数类（无需手动指定）
    # class_counts = np.bincount(y)
    # minority_class = np.argmin(class_counts)
    # majority_class = np.argmax(class_counts)
    #
    # # 计算需生成的少数类样本数（确保目标数 >= 原少数类样本数）
    # n_minority = class_counts[minority_class]
    # n_majority = class_counts[majority_class]
    # target_count = max(n_majority, int(n_majority * 0.8))  # 至少与多数类数量相同
    #
    # # 避免SMOTE参数错误
    # if target_count <= n_minority:
    #     print(f"警告：少数类无需过采样 (原始={n_minority}, 目标={target_count})")
    #     return X, y
    #
    # # 应用SMOTE
    # smote = SMOTE(
    #     sampling_strategy={minority_class: target_count},
    #     k_neighbors=min(5, n_minority - 1)  # 防止邻居数超过样本数
    # )
    # try:
    #     X_res, y_res = smote.fit_resample(X, y)
    # except ValueError as e:
    #     print(f"SMOTE错误: {e}")
    #     return X, y
    #
    # return X_res, y_res

    assert len(np.unique(y)) == 2, "balance_data should only be applied to training data!"

    # 获取原始数据类别分布
    class_counts = np.bincount(y)
    minority_class = np.argmin(class_counts)  # 少数类标签（例如1）
    majority_class = np.argmax(class_counts)  # 多数类标签（例如0）

    # 设置过采样目标为多数类数量的80%（避免完全平衡导致过拟合）
    target_count = int(class_counts[majority_class] * 0.8)

    # 仅对少数类过采样
    smote = SMOTE(
        sampling_strategy={minority_class: target_count},
        k_neighbors=5
    )
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res


# 检查并转换标签格式
def convert_labels(y):
    # 处理字节字符串（如 ARFF 文件中的 b'buggy' 和 b'clean'）
    if isinstance(y[0], bytes):
        y = [label.decode('utf-8').strip().lower() for label in y]  # 解码并去除首尾空格

    # 映射标签到 0/1
    y_converted = []
    for label in y:
        if isinstance(label, str):
            # 明确映射逻辑
            label_clean = label.strip().lower()
            if label_clean in {'buggy', 'defect', 'true', '1', 'Y', 'y'}:
                y_converted.append(1)
            elif label_clean in {'clean', 'non-defect', 'false', '0', 'N', 'n'}:
                y_converted.append(0)
            else:
                raise ValueError(f"无法识别的标签值: '{label}' (原始值: {label})")
        else:
            y_converted.append(int(label))

    # 转换为整数数组并验证
    y_array = np.array(y_converted, dtype=np.int64)
    unique_labels = np.unique(y_array)
    if len(unique_labels) < 2:
        print(f"错误: 标签转换后仅包含 {unique_labels}。原始标签示例: {y[:5]}")


    return y_array

