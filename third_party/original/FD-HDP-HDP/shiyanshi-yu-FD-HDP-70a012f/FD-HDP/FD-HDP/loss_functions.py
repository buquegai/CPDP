# loss_functions.py

import torch
import torch.nn as nn

class LossFunctions:
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        self.alpha = alpha  # 重建损失权重
        self.beta = beta    # 对抗损失权重
        self.gamma = gamma  # 正交损失权重
        self.cross_entropy_loss = nn.BCELoss()
    #缺陷损失函数，计算预测值和真实值之间的二元交叉熵损失
    def defect_loss(self, y_pred, y_true):
        # 将预测值进行维度压缩，去除多余的维度，确保与真实值维度匹配
        # 并将真实值转换为浮点数类型
        return self.cross_entropy_loss(y_pred.squeeze(), y_true.float())

    # 重建损失函数，计算原始特征和重构特征之间的均方误差
    def reconstruction_loss(self, e_original, e_reconstructed):
        # 确保 e_reconstructed 的维度与 e_original 一致
        if e_original.size(1) != e_reconstructed.size(1):  # 如果特征维度不匹配
            e_reconstructed = e_reconstructed.view(e_original.size(0), -1)  # 调整维度

        # 计算均方误差，即 (原始特征 - 重构特征) 的平方的均值
        return torch.mean((e_original - e_reconstructed) ** 2)

    # 领域对抗损失函数，计算领域预测值和领域标签之间的二元交叉熵损失
    def domain_adversarial_loss(self, domain_pred, domain_labels):
        # 将领域预测值和领域标签进行维度压缩，去除多余的维度
        # 并将领域标签转换为浮点数类型
        return self.cross_entropy_loss(domain_pred.squeeze(), domain_labels.float().squeeze())

    #正交性损失函数，计算共享特征和相关特征之间的正交性损失
    def orthogonality_loss(self, h_shared_source, h_related_source, h_shared_target, h_related_target):



        # def _frobenius_norm(a, b):
        #     correlation = torch.mm(a.t(), b)  # (feature_dim, feature_dim)
        #     return torch.norm(correlation, p='fro') ** 2  # 论文公式 (11)

        # loss_source = _frobenius_norm(h_shared_source, h_related_source)
        # loss_target = _frobenius_norm(h_shared_target, h_related_target)
        # return (loss_source + loss_target) / (h_shared_source.size(0) ** 2)  # 归一化
        # loss_source = _frobenius_norm(h_shared_source, h_related_source) / h_shared_source.size(0)
        # loss_target = _frobenius_norm(h_shared_target, h_related_target) / h_shared_target.size(0)
        # return (loss_source + loss_target) / (h_shared_source.size(1))


        # 计算 F-范数（论文公式11）
        loss_source = torch.norm(torch.mm(h_shared_source.t(), h_related_source), p='fro') ** 2
        loss_target = torch.norm(torch.mm(h_shared_target.t(), h_related_target), p='fro') ** 2
        # return loss_source + loss_target
        return (loss_source + loss_target) / (h_shared_source.size(1))
        # # 计算余弦相似度
        # def _cos_sim(a, b):
        #     a_norm = a / (torch.norm(a, dim=1, keepdim=True) + 1e-8)
        #     b_norm = b / (torch.norm(b, dim=1, keepdim=True) + 1e-8)
        #     return torch.mean(torch.abs(torch.sum(a_norm * b_norm, dim=1)))
        #
        # loss_source = _cos_sim(h_shared_source, h_related_source)
        # loss_target = _cos_sim(h_shared_target, h_related_target)
        # return loss_source + loss_target


