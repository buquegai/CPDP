from __future__ import annotations

import numpy as np


def select_diverse_indices(
    values: np.ndarray,
    count: int,
    seed: int,
    pool_limit: int,
) -> np.ndarray:
    """Greedy farthest-point selection over a bounded candidate pool."""

    candidates = np.asarray(values, dtype=np.float64)
    if count <= 0 or len(candidates) == 0:
        return np.empty(0, dtype=np.int64)
    if len(candidates) <= count:
        return np.arange(len(candidates), dtype=np.int64)
    random = np.random.default_rng(seed)
    effective_limit = max(count, pool_limit)
    if len(candidates) > effective_limit:
        pool_indices = np.sort(
            random.choice(len(candidates), size=effective_limit, replace=False)
        )
    else:
        pool_indices = np.arange(len(candidates), dtype=np.int64)
    pool = candidates[pool_indices]
    center = np.median(pool, axis=0)
    first = int(np.argmin(np.sum((pool - center) ** 2, axis=1)))
    selected = np.empty(count, dtype=np.int64)
    selected[0] = first
    minimum_distance = np.full(len(pool), np.inf, dtype=np.float64)
    for position in range(1, count):
        previous = pool[selected[position - 1]]
        distances = np.sum((pool - previous) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, distances)
        minimum_distance[selected[:position]] = -np.inf
        selected[position] = int(np.argmax(minimum_distance))
    return pool_indices[selected]
