# main.py
import copy

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from torch import optim, nn

from data_preprocessing import load_arff, balance_data, convert_labels, MLPFeatureCompression
from feature_disentanglement import FeatureDisentanglement, FeatureReconstructor, DomainSharedClassifier, \
    GradientReversalLayer
from prediction_layer import Predictor
from loss_functions import LossFunctions
from train import pretrain_model, fine_tune_model
from evaluate import evaluate_model
import torch
import numpy as np
import torch.nn.functional as F

# 加载和处理数据
source_data = load_arff('dataset/PROMISE/camel-1.6.arff')  # 源领域数据
target_data = load_arff('dataset/ReLink/Safe.arff')   # 目标领域数据

X_source_raw = source_data.drop('defects', axis=1).values  # 原始特征
y_source_raw = convert_labels(source_data['defects'].values)
X_target_raw = target_data.drop('defects', axis=1).values  # 原始特征
y_target_raw = convert_labels(target_data['defects'].values)

# print("原始源域类别分布:", np.unique(y_source_raw, return_counts=True))
# print("原始目标域类别分布:", np.unique(y_target_raw, return_counts=True))


# 2. 先平衡数据（在原始特征空间进行SMOTE）
X_source_balanced, y_source_balanced = X_source_raw, y_source_raw
# X_source_balanced, y_source_balanced = balance_data(X_source_raw, y_source_raw)
X_target_balanced, y_target_balanced = balance_data(X_target_raw, y_target_raw)

# 3. 归一化平衡后的数据
X_source_normalized = MinMaxScaler().fit_transform(X_source_balanced)  # 对平衡后的数据归一化
X_target_normalized = MinMaxScaler().fit_transform(X_target_balanced)

# 4. 特征压缩
source_compressor = MLPFeatureCompression(input_dim=X_source_raw.shape[1], compressed_dim=128)
target_compressor = MLPFeatureCompression(input_dim=X_target_raw.shape[1], compressed_dim=128)
X_source_compressed = source_compressor(torch.tensor(X_source_normalized).float()).detach().numpy()
X_target_compressed = target_compressor(torch.tensor(X_target_normalized).float()).detach().numpy()


# print("平衡后源域类别分布:", np.unique(y_source_balanced, return_counts=True))
# print("平衡后目标域类别分布:", np.unique(y_target_balanced, return_counts=True))
# print("压缩后特征均值:", X_source_compressed.mean(), "方差:", X_source_compressed.var())

# 6. 划分目标域为标记/未标记数据（基于压缩后的平衡数据）
X_target_labeled, X_target_unlabeled, y_target_labeled, _ = train_test_split(
    X_target_compressed,
    y_target_balanced,
    test_size=0.5,
    stratify=y_target_balanced
)

# 7. 转换为Tensor
X_source_res = torch.tensor(X_source_compressed).float()
y_source_res = torch.tensor(y_source_balanced).float()
X_target_labeled = torch.tensor(X_target_labeled).float()
X_target_unlabeled = torch.tensor(X_target_unlabeled).float()
y_target_labeled = torch.tensor(y_target_labeled).float()

# 全局初始化所有模型组件
source_compressor = MLPFeatureCompression(input_dim=X_source_raw.shape[1], compressed_dim=128)
target_compressor = MLPFeatureCompression(input_dim=X_target_raw.shape[1], compressed_dim=128)
disentanglement_layer = FeatureDisentanglement(d_in=128, hidden_dims=[64, 32], disentangle_dim=64)
feature_reconstructor = FeatureReconstructor(d_in=128, h_specific_dim=128, h_share_dim=128)
domain_shared_classifier = DomainSharedClassifier(disentangle_dim=128)
prediction_layer = Predictor(d_in=128, hidden_dims=[128, 64], alpha_T=0.5, alpha_S=0.5)
loss_functions = LossFunctions(alpha=1.0, beta=1.0, gamma=1.0)  # 显式设置权重

# 加载目标域全部数据（包含未标记部分）
X_target_full = torch.cat([X_target_labeled, X_target_unlabeled])  # 合并已标记和未标记数据
y_target_full = torch.cat([y_target_labeled, torch.zeros(len(X_target_unlabeled))])  # 伪标签


# 进行预训练
pretrain_model(
    X_source_res,
    X_target_full,
    disentanglement_layer,
    feature_reconstructor,
    domain_shared_classifier,
    loss_functions,
    epochs=50
)

with torch.no_grad():
    # 提取源域和目标域的共享特征
    _, _, h_shared_source, _ = disentanglement_layer(X_source_res, X_source_res)
    _, _, h_shared_target, _ = disentanglement_layer(X_target_labeled, X_target_labeled)

    # 计算余弦相似度
    similarity = torch.cosine_similarity(
        h_shared_source.mean(dim=0).unsqueeze(0),
        h_shared_target.mean(dim=0).unsqueeze(0)
    ).item()
    # print(f"共享特征相似度: {similarity:.4f}")  # 预期接近1.0



def train_local_model(model, X_train, y_train, epochs=5):
    """短期微调预测层参数"""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for _ in range(epochs):
        optimizer.zero_grad()
        # 获取特征（冻结解耦层）
        with torch.no_grad():
            _, h_rel, _, h_shr = disentanglement_layer(X_source_res, X_train)
        # 预测
        y_shared = model.domain_shared(h_shr)
        y_specific = model.domain_specific_T(h_rel)
        y_pred = model.alpha_T * y_shared + (1 - model.alpha_T) * y_specific
        # 计算损失
        loss = F.binary_cross_entropy_with_logits(y_pred.squeeze(), y_train)
        loss.backward()
        optimizer.step()


def select_alpha(X_target_labeled, y_target_labeled):
    best_alpha, best_f1 = 0.5, 0.0
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for alpha in np.linspace(0, 1, 11):  # 0.0, 0.1, ..., 1.0
        fold_f1 = []

        for train_idx, val_idx in kf.split(X_target_labeled):
            X_train, X_val = X_target_labeled[train_idx], X_target_labeled[val_idx]
            y_train, y_val = y_target_labeled[train_idx], y_target_labeled[val_idx]

            # 克隆模型（避免污染）
            clone_model = copy.deepcopy(prediction_layer)
            clone_model.alpha_T = alpha

            # 短期微调（仅目标域训练数据）
            optimizer = optim.Adam(clone_model.parameters(), lr=0.001)
            for _ in range(10):
                optimizer.zero_grad()
                with torch.no_grad():
                    _, h_rel, _, h_shr = disentanglement_layer(X_source_res, X_train)
                y_pred = clone_model(h_rel, h_rel, h_shr, h_shr)[0]
                loss = F.binary_cross_entropy_with_logits(y_pred.squeeze(), y_train)
                loss.backward()
                optimizer.step()

            # 验证集评估
            with torch.no_grad():
                _, h_val_rel, _, h_val_shr = disentanglement_layer(X_source_res, X_val)
                y_prob = torch.sigmoid(clone_model.alpha_T * clone_model.domain_shared(h_val_shr) +
                                       (1 - clone_model.alpha_T) * clone_model.domain_specific_T(h_val_rel))
                y_pred = (y_prob >= 0.5).float()
                f1 = f1_score(y_val.numpy(), y_pred.numpy())
                fold_f1.append(f1)

        avg_f1 = np.mean(fold_f1)
        if avg_f1 > best_f1:
            best_alpha, best_f1 = alpha, avg_f1

    return best_alpha

# 应用选择结果
best_alpha_T = select_alpha(X_target_labeled, y_target_labeled)
prediction_layer.alpha_T = best_alpha_T
# prediction_layer.alpha_S = 0.5

# 微调模型
fine_tune_model(
    X_source_res, y_source_res,
    X_target_labeled, y_target_labeled,
    disentanglement_layer,
    feature_reconstructor,
    domain_shared_classifier,
    prediction_layer,
    loss_functions,
    epochs=100
)
# 评估模型
evaluate_model(disentanglement_layer, prediction_layer, loss_functions, X_source_res, X_target_labeled, y_target_labeled)
