from __future__ import annotations


def sample_feature_mask(torch, values, minimum_ratio: float, maximum_ratio: float):
    batch_size, feature_dim = values.shape
    ratios = torch.empty(batch_size, 1, device=values.device).uniform_(
        minimum_ratio, maximum_ratio
    )
    mask = torch.rand_like(values) < ratios
    if feature_dim > 1:
        empty_rows = ~mask.any(dim=1)
        if empty_rows.any():
            columns = torch.randint(feature_dim, (int(empty_rows.sum()),), device=values.device)
            mask[empty_rows, columns] = True
        full_rows = mask.all(dim=1)
        if full_rows.any():
            columns = torch.randint(feature_dim, (int(full_rows.sum()),), device=values.device)
            mask[full_rows, columns] = False
    return mask


def latent_prediction_loss(torch, predicted, target):
    functional = torch.nn.functional
    predicted = functional.normalize(predicted, dim=1)
    target = functional.normalize(target.detach(), dim=1)
    return functional.mse_loss(predicted, target)


def variance_covariance_losses(
    torch,
    representations,
    minimum_std: float = 1.0,
    include_covariance: bool = True,
):
    if representations.shape[0] < 2:
        zero = representations.sum() * 0.0
        return zero, zero
    centered = representations - representations.mean(dim=0, keepdim=True)
    standard_deviation = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
    variance_loss = torch.relu(minimum_std - standard_deviation).mean()
    if not include_covariance:
        return variance_loss, variance_loss.new_zeros(())
    covariance = centered.T @ centered / max(1, representations.shape[0] - 1)
    diagonal = torch.diagonal(covariance)
    covariance_loss = (covariance.square().sum() - diagonal.square().sum()) / max(
        1, representations.shape[1]
    )
    return variance_loss, covariance_loss


def weighted_coral_loss(torch, source, target, source_weights):
    if source.shape[0] < 2 or target.shape[0] < 2:
        return source.sum() * 0.0
    weights = source_weights / source_weights.sum().clamp_min(1e-8)
    source_mean = (source * weights.unsqueeze(1)).sum(dim=0, keepdim=True)
    source_centered = source - source_mean
    source_denominator = (1.0 - weights.square().sum()).clamp_min(1e-6)
    source_covariance = (
        source_centered.T @ (source_centered * weights.unsqueeze(1))
    ) / source_denominator
    target_centered = target - target.mean(dim=0, keepdim=True)
    target_covariance = target_centered.T @ target_centered / max(
        1, target.shape[0] - 1
    )
    latent_dim = max(1, source.shape[1])
    return (source_covariance - target_covariance).square().sum() / (
        4.0 * latent_dim * latent_dim
    )
