"""风机间距约束。"""

import numpy as np


def check_min_spacing(
    positions: np.ndarray,
    min_distance: float,
) -> tuple[bool, np.ndarray]:
    """检查所有风机对之间的间距是否满足最小距离要求。

    Parameters
    ----------
    positions : np.ndarray
        风机位置，形状为 (N_turbines, 2)
    min_distance : float
        最小允许间距 (m)

    Returns
    -------
    tuple[bool, np.ndarray]
        - 是否所有间距都满足要求
        - 不满足要求的风机对索引数组，形状为 (M, 2)，M 为违规对数
    """
    positions = np.asarray(positions, dtype=np.float64)
    n = positions.shape[0]
    violations = []

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < min_distance - 1e-9:
                violations.append([i, j])

    if violations:
        return False, np.array(violations, dtype=int)
    else:
        return True, np.zeros((0, 2), dtype=int)


def compute_min_spacing_from_diameters(
    rotor_diameters: np.ndarray,
    min_multiple: float = 5.0,
) -> float:
    """根据转子直径计算最小间距（取最大直径的倍数）。"""
    return float(min_multiple * np.max(rotor_diameters))


def compute_pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """计算所有风机对之间的距离矩阵，形状为 (N, N)。"""
    positions = np.asarray(positions, dtype=np.float64)
    n = positions.shape[0]
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            dist[i, j] = d
            dist[j, i] = d
    return dist


def enforce_min_spacing(
    positions: np.ndarray,
    min_distance: float,
    domain,
    rng: np.random.Generator | None = None,
    max_iterations: int = 300,
    time_budget_s: float = 10.0,
) -> np.ndarray:
    """在给定可行域（租赁边界扣除禁建区）内修复最小间距约束。

    Parameters
    ----------
    positions : np.ndarray
        初始风机位置，形状为 (N, 2)
    min_distance : float
        最小间距 (m)
    domain : FeasibleDomain
        可行域，必须提供 ``repair_layout`` / ``validate_layout``。
    rng : np.random.Generator | None
        随机数生成器
    max_iterations : int
        最大修复迭代次数
    time_budget_s : float
        修复时间预算（秒）

    Returns
    -------
    np.ndarray
        同时满足间距、租赁边界与全部禁建净距的位置 (N, 2)

    Raises
    ------
    InfeasibleLayoutError
        有限时间内无法修复时抛出，携带具体约束原因。
    """
    return domain.repair_layout(
        np.asarray(positions, dtype=np.float64),
        min_spacing=float(min_distance),
        rng=rng,
        max_iterations=max_iterations,
        time_budget_s=time_budget_s,
    )
