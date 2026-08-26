#prediction_layer.py
import torch
import torch.nn as nn

# 领域相关预测器
class DomainSpecificPredictor(nn.Module):
    def __init__(self, d_in, hidden_dims):
        super(DomainSpecificPredictor, self).__init__()
        layers = []
        prev_dim = d_in
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))  # 最后一层，输出一个概率值
        layers.append(nn.Sigmoid())   # 使用 Sigmoid 激活函数输出概率
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# 领域共享预测器
class DomainSharedPredictor(nn.Module):
    def __init__(self, d_in, hidden_dims):
        super(DomainSharedPredictor, self).__init__()
        layers = []
        prev_dim = d_in
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))  # 最后一层，输出一个概率值
        layers.append(nn.Sigmoid())  # 使用 Sigmoid 激活函数输出概率
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# 完整的预测器
class Predictor(nn.Module):
    def __init__(self, d_in=128, hidden_dims=None, alpha_T=0.5, alpha_S=0.5):
        super(Predictor, self).__init__()
        # 目标域领域相关预测器
        if hidden_dims is None:
            hidden_dims = [128, 64]
        self.domain_specific_T = DomainSpecificPredictor(d_in=128, hidden_dims=hidden_dims)
        # 源域领域相关预测器
        self.domain_specific_S = DomainSpecificPredictor(d_in=128, hidden_dims=hidden_dims)
        # 领域共享预测器
        self.domain_shared = DomainSharedPredictor(d_in=128, hidden_dims=hidden_dims)
        # 权重参数
        self.alpha_T = alpha_T
        self.alpha_S = alpha_S

    def forward(self, h_specific_T, h_specific_S, h_share_T, h_share_S):
        # 目标域预测
        y_specific_T = self.domain_specific_T(h_specific_T)
        y_share_T = self.domain_shared(h_share_T)
        y_T = self.alpha_T * y_share_T + (1 - self.alpha_T) * y_specific_T  # 加权求和

        # 源域预测
        y_specific_S = self.domain_specific_S(h_specific_S)
        y_share_S = self.domain_shared(h_share_S)
        y_S = self.alpha_S * y_share_S + (1 - self.alpha_S) * y_specific_S  # 加权求和

        return y_T, y_S
