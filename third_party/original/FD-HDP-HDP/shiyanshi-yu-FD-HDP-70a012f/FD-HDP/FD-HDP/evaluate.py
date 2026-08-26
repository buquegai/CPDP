# evaluate.py
import copy

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold


def evaluate_model(disentanglement_layer, prediction_layer, loss_functions, X_source, X_target, y_target):

    kf = RepeatedStratifiedKFold(n_splits=3, n_repeats=10,  random_state=42)
    # 初始化评估容器
    metrics = {
        'precision': [], 'recall': [], 'f1': [], 'auc': [], 'g_mean': [], 'h_mean': []
    }
    threshold_history = []  # 记录每折最优阈值
    # 遍历每个交叉验证折叠
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_target, y_target)):
        X_train_fold, X_test_fold = X_target[train_idx], X_target[test_idx]
        y_train_fold, y_test_fold = y_target[train_idx], y_target[test_idx]

        # # 调试输出：标签分布
        print(f"\n=== Fold {fold_idx + 1} ===")
        print("测试集标签分布:", np.unique(y_test_fold.numpy(), return_counts=True))

        # 克隆预测层并微调
        cloned_pred_layer = copy.deepcopy(prediction_layer)
        optimizer = torch.optim.Adam(cloned_pred_layer.parameters(), lr=0.001)

        # 微调10个epoch
        for epoch in range(10):
            optimizer.zero_grad()
            with torch.no_grad():
                _, h_train_rel, _, h_train_shr = disentanglement_layer(X_source, X_train_fold)
            y_shared = cloned_pred_layer.domain_shared(h_train_shr)
            y_specific = cloned_pred_layer.domain_specific_T(h_train_rel)
            y_pred = cloned_pred_layer.alpha_T * y_shared + (1 - cloned_pred_layer.alpha_T) * y_specific
            loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred.squeeze(), y_train_fold)
            loss.backward()
            optimizer.step()

        # ==== 动态选择阈值 ====
        with torch.no_grad():
            # 在验证集上选择阈值（此处使用训练集作为验证集简化实现）
            _, h_val_rel, _, h_val_shr = disentanglement_layer(X_source, X_train_fold)
            y_shared_val = cloned_pred_layer.domain_shared(h_val_shr)
            y_specific_val = cloned_pred_layer.domain_specific_T(h_val_rel)
            y_prob_val = torch.sigmoid(cloned_pred_layer.alpha_T * y_shared_val +
                                       (1 - cloned_pred_layer.alpha_T) * y_specific_val)

            # 搜索最佳阈值（0.3-0.7，步长0.05）
            best_thresh = 0.5
            # best_f1 = 0
            best_gmean = 0
            if len(np.unique(y_train_fold)) < 2:
                best_thresh = 0.5  # 使用默认阈值
            else:
                # for thresh in np.arange(0.3, 0.71, 0.05):
                #     y_pred_val = (y_prob_val >= thresh).float()
                #     f1 = f1_score(y_train_fold.numpy(), y_pred_val.numpy())
                #     if f1 > best_f1:
                #         best_f1 = f1
                #         best_thresh = thresh
                # threshold_history.append(best_thresh)  # 记录阈值
                for thresh in np.linspace(0.1, 0.9, 50):
                    y_pred_val = (y_prob_val >= thresh).float()
                    tn, fp, fn, tp = confusion_matrix(y_train_fold.numpy(), y_pred_val.numpy()).ravel()
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    sensitivity = tp / (tp + fn)
                    gmean = np.sqrt(specificity * sensitivity)
                    if gmean > best_gmean:
                        best_gmean = gmean
                        best_thresh = thresh
        # ==== 使用最佳阈值评估测试集 ====
        with torch.no_grad():
            _, h_test_rel, _, h_test_shr = disentanglement_layer(X_source, X_test_fold)
            y_shared_test = cloned_pred_layer.domain_shared(h_test_shr)
            y_specific_test = cloned_pred_layer.domain_specific_T(h_test_rel)
            y_prob_test = torch.sigmoid(cloned_pred_layer.alpha_T * y_shared_test +
                                        (1 - cloned_pred_layer.alpha_T) * y_specific_test)
            y_pred_test = (y_prob_test >= best_thresh).float().cpu().numpy()
            y_true_test = y_test_fold.cpu().numpy()


            # ==== 计算指标 ====
            if np.sum(y_pred_test) == 0:  # 处理全负预测
                precision = 0.0
                recall = 0.0
            else:
                precision = precision_score(y_true_test, y_pred_test, zero_division=0)
                recall = recall_score(y_true_test, y_pred_test, zero_division=0)
            f1 = f1_score(y_true_test, y_pred_test)
            auc = roc_auc_score(y_true_test, y_prob_test.numpy())

            # 计算混淆矩阵（确保顺序正确）

            if len(np.unique(y_true_test)) < 2:
                print(f"警告：Fold {fold_idx + 1}的测试集仅包含单一类别，跳过混淆矩阵计算")
                tn, fp, fn, tp = 0, 0, 0, 0  # 根据实际情况处理或跳过指标
            else:
                tn, fp, fn, tp = confusion_matrix(y_true_test, y_pred_test).ravel()


            # tn, fp, fn, tp = confusion_matrix(y_true_test, y_pred_test).ravel()
            print(f"混淆矩阵: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)
            metrics['auc'].append(auc)
            metrics['g_mean'].append(
                np.sqrt(sensitivity * specificity) if (sensitivity > 0 and specificity > 0) else 0.0)
            metrics['h_mean'].append(
                        2 * sensitivity * specificity / (sensitivity + specificity) if (sensitivity + specificity) > 0 else 0)

    # ==== 打印结果 ====
    print("\n=== 最终评估结果 ===")
    print(f"Precision: {np.mean(metrics['precision']):.4f}")
    print(f"Recall:    {np.mean(metrics['recall']):.4f}")
    print(f"F1-Score:  {np.mean(metrics['f1']):.4f}")
    print(f"AUC:       {np.mean(metrics['auc']):.4f}")
    print(f"g_mean:    {np.mean(metrics['g_mean']):.4f}")
    print(f"h_mean:    {np.mean(metrics['h_mean']):.4f}")

    # 输出阈值分布
    # print(f"\n阈值分布：均值={np.mean(threshold_history):.2f}, 范围[{np.min(threshold_history):.2f}, {np.max(threshold_history):.2f}]")

    return metrics
