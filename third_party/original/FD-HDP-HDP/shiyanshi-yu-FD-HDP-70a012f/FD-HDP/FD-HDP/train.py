# train.py

import torch
import torch.optim as optim

#预训练
def pretrain_model(X_source, X_target, disentanglement_layer, feature_reconstructor, domain_shared_classifier, loss_functions, epochs=100):

    # 定义优化器，使用 Adam 优化器，学习率为 0.001
    optimizer = optim.Adam(list(disentanglement_layer.parameters()) +
                           list(feature_reconstructor.parameters()) +
                           list(domain_shared_classifier.parameters()), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    #开始循环
    for epoch in range(epochs):
        #梯度清零
        optimizer.zero_grad()
        #得到各相关特征
        h_source_related, h_target_related, h_source_shared, h_target_shared = disentanglement_layer(
            X_source, X_target)
        # 使用 FeatureReconstructor 进行特征重构
        reconstructed_source = feature_reconstructor(h_source_shared, h_source_related)
        reconstructed_target = feature_reconstructor(h_target_shared, h_target_related)

        #计算重建损失
        loss_reconstruction_source = loss_functions.reconstruction_loss(X_source, reconstructed_source)
        loss_reconstruction_target = loss_functions.reconstruction_loss(X_target, reconstructed_target)

        domain_pred_source = domain_shared_classifier(h_source_shared)
        domain_pred_target = domain_shared_classifier(h_target_shared)

        domain_labels_source = torch.zeros(X_source.size(0), dtype=torch.float32)
        domain_labels_target = torch.ones(X_target.size(0), dtype=torch.float32)

        loss_domain = loss_functions.domain_adversarial_loss(
            torch.cat([domain_pred_source, domain_pred_target]),
            torch.cat([domain_labels_source, domain_labels_target])
        )

        # 正交性损失
        loss_orth = loss_functions.orthogonality_loss(
            h_source_shared, h_source_related,
            h_target_shared, h_target_related
        )

        # 总损失（公式17）
        total_loss = loss_reconstruction_source + loss_reconstruction_target + loss_domain + loss_orth

        domain_acc_source = (domain_pred_source.round() == 0).float().mean()
        domain_acc_target = (domain_pred_target.round() == 1).float().mean()
        # print(f"领域分类准确率: 源域={domain_acc_source.item():.4f}, 目标域={domain_acc_target.item():.4f}")
        #
        # print(f"预训练损失: 重构={loss_reconstruction_source + loss_reconstruction_target:.4f}, 对抗={loss_domain:.4f}, 正交={loss_orth:.4f}")
        #
        # print(f"正交损失（归一化后）: {loss_orth.item():.4f}")  # 预期逐步下降至 < 100.0
        print(
            f"Epoch [{epoch}/{epochs}], "
            f"重构损失: {loss_reconstruction_source + loss_reconstruction_target:.4f}, "
            f"对抗损失: {loss_domain:.4f}, "
            f"正交损失: {loss_orth:.4f}"
        )

        total_loss.backward()       #反向传播计算梯度

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            list(disentanglement_layer.parameters()) +
            list(feature_reconstructor.parameters()) +
            list(domain_shared_classifier.parameters()),
            max_norm=1.0  # 最大梯度范数限制为1.0
        )

        # 打印梯度信息
        total_norm = 0
        for p in disentanglement_layer.parameters():
            if p.grad is not None:
                total_norm += p.grad.detach().data.norm(2).item() ** 2
        # print(f"梯度范数: {total_norm ** 0.5:.4f}")


        optimizer.step()
        scheduler.step()
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{epochs}], Pre-training Loss: {total_loss.item():.4f}')



#微调
def fine_tune_model(X_source, y_source, X_target_labeled, y_target_labeled, disentanglement_layer, feature_reconstructor, domain_shared_classifier, prediction_layer, loss_functions, epochs=100):

    # 定义优化器，使用 Adam 优化器，学习率为 0.001
    optimizer = optim.Adam(list(disentanglement_layer.parameters()) +
                           list(feature_reconstructor.parameters()) +
                           list(domain_shared_classifier.parameters()) +
                           list(prediction_layer.parameters()), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # 每30轮学习率×0.1

    #开始循环
    for epoch in range(epochs):
        # 梯度清零
        optimizer.zero_grad()
        # # 得到相关特征及预测结果
        h_source_related, h_target_related, h_source_shared, h_target_shared = disentanglement_layer(X_source, X_target_labeled)

        # 预测结果
        y_T, y_S = prediction_layer(
            h_target_related, h_source_related,
            h_target_shared, h_source_shared
        )

        # 计算缺陷损失
        loss_defect = loss_functions.defect_loss(y_S, y_source) + loss_functions.defect_loss(y_T, y_target_labeled)


        # 重构损失
        reconstructed_source = feature_reconstructor(h_source_related, h_source_shared)
        reconstructed_target = feature_reconstructor(h_target_related, h_target_shared)
        loss_rec = loss_functions.reconstruction_loss(X_source, reconstructed_source) + \
                   loss_functions.reconstruction_loss(X_target_labeled, reconstructed_target)


        # 对抗损失
        domain_pred_source = domain_shared_classifier(h_source_shared)
        domain_pred_target = domain_shared_classifier(h_target_shared)
        loss_domain = loss_functions.domain_adversarial_loss(
            torch.cat([domain_pred_source, domain_pred_target]),
            torch.cat([torch.zeros_like(y_source), torch.ones_like(y_target_labeled)])
        )


        # 正交损失
        loss_orth = loss_functions.orthogonality_loss(
            h_source_shared, h_source_related,
            h_target_shared, h_target_related
        )

       # 总损失（公式12）
        total_loss = loss_defect + loss_rec + loss_domain + loss_orth

        total_loss.backward()       #反向传播计算梯度
        optimizer.step()            #根据梯度更新参数
        scheduler.step()

        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{epochs}], Fine-tuning Loss: {total_loss.item():.4f}')
