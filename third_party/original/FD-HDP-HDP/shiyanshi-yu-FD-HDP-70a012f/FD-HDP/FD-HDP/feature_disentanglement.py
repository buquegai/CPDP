# feature_disentanglement.py

import torch
import torch.nn as nn

# 领域相关特征提取器
class DomainSpecificFeatureExtractor(nn.Module):
    def __init__(self, d_in, hidden_dims):
        super(DomainSpecificFeatureExtractor, self).__init__()
        layers = []
        prev_dim = d_in
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, d_in))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# 领域共享特征提取器
class DomainSharedFeatureExtractor(nn.Module):
    def __init__(self, d_in, hidden_dims):
        super(DomainSharedFeatureExtractor, self).__init__()
        layers = []
        prev_dim = d_in
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, d_in))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# 特征解耦器
class FeatureDisentanglement(nn.Module):
    def __init__(self, d_in=128, hidden_dims=None, disentangle_dim=64):
        super().__init__()
        # 领域相关特征提取器
        if hidden_dims is None:
            hidden_dims = [256, 128]
        self.domain_specific_T = DomainSpecificFeatureExtractor(d_in, hidden_dims)
        self.domain_specific_S = DomainSpecificFeatureExtractor(d_in, hidden_dims)
        # 领域共享特征提取器
        self.domain_shared = DomainSharedFeatureExtractor(d_in, hidden_dims)

    def forward(self, e_T, e_S):
        assert e_T.shape[1] == e_S.shape[1], "Source/Target feature dimensions must match"
        h_specific_T = self.domain_specific_T(e_T)
        h_specific_S = self.domain_specific_S(e_S)
        h_shared_T = self.domain_shared(e_T)
        h_shared_S = self.domain_shared(e_S)
        return h_specific_T, h_specific_S, h_shared_T, h_shared_S

#特征重构器
class FeatureReconstructor(nn.Module):
    def __init__(self, d_in=128, h_specific_dim=128, h_share_dim=128):
        super().__init__()
        self.fc = nn.Linear(h_specific_dim + h_share_dim, d_in)
        self.relu = nn.ReLU()

    def forward(self, h_specific, h_share):
        combined = torch.cat((h_specific, h_share), dim=1)
        return self.relu(self.fc(combined))

# 领域分类器
# 梯度反转层
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class DomainSharedClassifier(nn.Module):
    def __init__(self, disentangle_dim):
        super().__init__()
        self.fc = nn.Linear(disentangle_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h_share):
        h_share = GradientReversalLayer.apply(h_share, 1.0)
        return self.sigmoid(self.fc(h_share))

