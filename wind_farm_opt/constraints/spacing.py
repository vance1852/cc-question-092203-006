"""风机间距约束。"""

import numpy as np

from .geofence import Geofence, InfeasibleLayoutError


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
            if dist + 1e-9 < min_distance:
                violations.append([i, j])

    if violations:
        return False, np.array(violations, dtype=int)
    else:
        return True, np.zeros((0, 2), dtype=int)


def compute_min_spacing_from_diameters(
    rotor_diameters: np.ndarray,
    min_multiple: float = 5.0,
) -> float:
    """根据转子直径计算最小间距（取最大直径的倍数）。

    Parameters
    ----------
    rotor_diameters : np.ndarray
        每台风机的转子直径
    min_multiple : float
        最小间距倍数（相对于转子直径）

    Returns
    -------
    float
        最小间距 (m)
    """
    return float(min_multiple * np.max(rotor_diameters))


def compute_pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """计算所有风机对之间的距离矩阵。

    Parameters
    ----------
    positions : np.ndarray
        风机位置，形状为 (N, 2)

    Returns
    -------
    np.ndarray
        距离矩阵，形状为 (N, N)，对角线为 0
    """
    positions = np.asarray(positions, dtype=np.float64)
    n = positions.shape[0]
    delta = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    return np.linalg.norm(delta, axis=-1)


def enforce_min_spacing(
    positions: np.ndarray,
    min_distance: float,
    site: Geofence,
    rng: np.random.Generator | None = None,
    max_iterations: int = 120,
    max_relocation_rounds: int = 6,
    relocation_pool: int = 300,
) -> np.ndarray:
    """尝试通过移动风机同时满足最小间距与可行域（租赁边界 + 禁建净距）。

    修复分两个有限预算的阶段：

    1. 推开：对间距不足的风机对沿连线互斥，每步把越界点投影回可行域；
    2. 重定位：仍无法满足的风机从可行域候选点池中重新采样，并要求与
       其余风机保持间距。

    任一点无法投影回可行域，或预算耗尽后仍有违反，则抛出
    :class:`InfeasibleLayoutError`，其中携带具体约束诊断，绝不返回
    违规布局。最坏情况下的几何评估次数有明确上限
    （N × ``max_relocation_rounds`` × ``relocation_pool``）。

    Parameters
    ----------
    positions : np.ndarray
        初始风机位置，形状为 (N, 2)
    min_distance : float
        最小间距 (m)
    site : Geofence
        可行域（租赁边界与禁建区）
    rng : Optional[np.random.Generator]
        随机数生成器
    max_iterations : int
        推开阶段最大迭代次数
    max_relocation_rounds : int
        每台待重定位风机的候选点采样轮数
    relocation_pool : int
        每轮在可行域内采样的候选点数量

    Returns
    -------
    np.ndarray
        调整后的风机位置

    Raises
    ------
    InfeasibleLayoutError
        有限预算内无法得到可行布局。
    """
    if rng is None:
        rng = np.random.default_rng()

    positions = np.array(positions, dtype=np.float64, copy=True)
    n = positions.shape[0]

    def project(k: int) -> bool:
        """把第 k 台风机投影回可行域，返回是否成功。"""
        if site.is_feasible_point(positions[k]):
            return True
        try:
            positions[k] = site.project_into_feasible_region(positions[k], rng)
            return True
        except InfeasibleLayoutError:
            return False

    # 先把所有风机收回可行域
    for k in range(n):
        if not project(k):
            raise InfeasibleLayoutError(
                "间距修复失败：存在无法投影回可行域的风机（可能落入被禁建区"
                "净距完全覆盖的区域）",
                violations=site.diagnose(positions, min_distance),
            )

    # ---- 阶段 1：推开 ----
    for _ in range(max_iterations):
        valid, violations = check_min_spacing(positions, min_distance)
        if valid:
            return positions

        for i, j in violations:
            vec = positions[j] - positions[i]
            dist = np.linalg.norm(vec)
            if dist < 1e-12:
                vec = rng.standard_normal(2)
                dist = np.linalg.norm(vec)
            vec_norm = vec / dist

            push = (min_distance - dist) / 2.0 + 1e-6
            positions[i] -= vec_norm * push
            positions[j] += vec_norm * push

        for k in range(n):
            if not site.is_feasible_point(positions[k]):
                if not project(k):
                    break  # 交给阶段 2 / 最终诊断

    # ---- 阶段 2：重定位残留违规风机 ----
    valid, violations = check_min_spacing(positions, min_distance)
    if valid:
        return positions

    # ---- 阶段 2：重定位残留违规风机 ----
    # 每轮重新统计违规（重定位一台后冲突关系会变化），优先迁移冲突最多的
    # 风机；总迁移轮数有上限。
    relocated_any = True
    relocated_set: set[int] = set()
    for _ in range(n):
        valid, violations = check_min_spacing(positions, min_distance)
        if valid:
            break
        if not relocated_any:
            break

        bad_counts = np.zeros(n, dtype=int)
        for i, j in violations:
            bad_counts[i] += 1
            bad_counts[j] += 1
        idx = int(np.argmax(bad_counts))

        # 同一台风机反复冲突，说明候选池里已无空位
        if idx in relocated_set:
            break
        relocated_set.add(idx)

        others = np.where(np.arange(n) != idx)[0]
        relocated_any = False
        for _ in range(max_relocation_rounds):
            try:
                candidates = site.sample_feasible_points(
                    relocation_pool, rng, max_attempts=relocation_pool * 50
                )
            except InfeasibleLayoutError:
                break
            for cand in candidates:
                if all(
                    np.linalg.norm(cand - positions[k]) + 1e-9 >= min_distance
                    for k in others
                ):
                    positions[idx] = cand
                    relocated_any = True
                    break
            if relocated_any:
                break

        if not relocated_any:
            raise InfeasibleLayoutError(
                f"间距修复失败：风机 #{idx} 在可行域内找不到与其余风机保持 "
                f"{min_distance:.0f} m 间距的位置，场地可能过于拥挤",
                violations=site.diagnose(positions, min_distance),
            )

    valid, violations = check_min_spacing(positions, min_distance)
    feasible = site.feasible_mask(positions).all()
    if not valid or not feasible:
        raise InfeasibleLayoutError(
            "间距修复失败：推开与重定位后仍存在约束违反",
            violations=site.diagnose(positions, min_distance),
        )

    return positions
