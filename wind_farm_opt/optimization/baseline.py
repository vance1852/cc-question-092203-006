"""基线布局生成（规则网格）。

所有网格点与补点都必须落在可行域内（租赁边界扣除禁建区及其安全
净距）。无法在有限时间内容纳指定台数时抛出
:class:`InfeasibleLayoutError`，绝不返回带违规的布局。
"""

import numpy as np

from ..constraints.exclusion import FeasibleDomain, InfeasibleLayoutError
from ..constraints.spacing import (
    compute_min_spacing_from_diameters,
)


def _grid_points(
    domain: FeasibleDomain,
    n_turbines: int,
    min_spacing: float,
    staggered: bool,
) -> np.ndarray:
    """在可行域内铺设规则（或交错）网格点。"""
    n_rows = max(1, int(np.round(np.sqrt(n_turbines))))
    n_cols = max(1, int(np.ceil(n_turbines / n_rows)))

    x_min, x_max = domain.x_min, domain.x_max
    y_min, y_max = domain.y_min, domain.y_max

    margin = min_spacing * 0.5
    x_range = x_max - x_min - 2 * margin
    y_range = y_max - y_min - 2 * margin
    if x_range <= 0 or y_range <= 0:
        return np.zeros((0, 2), dtype=np.float64)

    spacing_x = min(x_range / max(n_cols - 1, 1), min_spacing * 1.5)
    spacing_y = min(y_range / max(n_rows - 1, 1), min_spacing * 1.5)
    spacing_x = max(spacing_x, min_spacing)
    spacing_y = max(spacing_y, min_spacing)

    start_x = x_min + margin + (x_range - spacing_x * (n_cols - 1)) / 2.0
    start_y = y_min + margin + (y_range - spacing_y * (n_rows - 1)) / 2.0

    positions: list[np.ndarray] = []
    for row in range(n_rows):
        offset = spacing_x / 2.0 if (staggered and row % 2 == 1) else 0.0
        for col in range(n_cols):
            pos = np.array([
                start_x + col * spacing_x + offset,
                start_y + row * spacing_y,
            ])
            if not domain.contains_point(pos):
                continue
            if all(np.linalg.norm(pos - p) >= min_spacing for p in positions):
                positions.append(pos)
    return np.array(positions, dtype=np.float64) if positions else np.zeros((0, 2))


def generate_grid_layout(
    domain: FeasibleDomain,
    n_turbines: int,
    rotor_diameters: np.ndarray,
    min_multiple: float = 5.0,
    rng: np.random.Generator | None = None,
    staggered: bool = False,
    time_budget_s: float = 20.0,
) -> np.ndarray:
    """生成规则网格基线，不足机位在可行域内随机补足。

    Parameters
    ----------
    domain : FeasibleDomain
        可行域（租赁边界 − 禁建区及净距）
    n_turbines : int
        风机台数
    rotor_diameters : np.ndarray
        每台风机转子直径
    min_multiple : float
        最小间距倍数（按最大转子直径）
    rng : np.random.Generator | None
        随机数生成器
    staggered : bool
        是否采用交错网格
    time_budget_s : float
        补点/修复的总时间预算（秒）

    Returns
    -------
    np.ndarray
        形状 (n_turbines, 2) 的可行机位

    Raises
    ------
    InfeasibleLayoutError
        可行域无法容纳该台数（携带具体约束原因）
    """
    if rng is None:
        rng = np.random.default_rng()

    min_spacing = compute_min_spacing_from_diameters(rotor_diameters, min_multiple)

    positions = _grid_points(domain, n_turbines, min_spacing, staggered)

    if positions.shape[0] < n_turbines:
        # 网格被禁建区切碎时，用可行域内带间距的拒绝采样补足，
        # 新点还要与已放置的网格点保持间距。
        needed = n_turbines - positions.shape[0]
        try:
            extra = domain.sample_separated_points(
                needed,
                min_spacing,
                rng,
                existing=positions,
                time_budget_s=time_budget_s,
            )
        except InfeasibleLayoutError as err:
            reasons = [
                f"规则网格仅能在可行域内布置 {positions.shape[0]}/{n_turbines} 台"
            ] + err.reasons
            raise InfeasibleLayoutError(
                f"基线布局失败：{err}", reasons=reasons, details=err.details
            ) from err
        positions = np.vstack([positions, extra])

    report = domain.validate_layout(positions, min_spacing)
    if not report.feasible:
        raise InfeasibleLayoutError(
            "基线布局生成后仍不满足约束",
            reasons=report.reasons,
            details={"reason": "baseline_invalid", "report": report.to_dict()},
        )
    return positions[:n_turbines]


def generate_staggered_grid_layout(
    domain: FeasibleDomain,
    n_turbines: int,
    rotor_diameters: np.ndarray,
    min_multiple: float = 5.0,
    dominant_direction: float = 270.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """生成交错网格基线（错位排列以减少主风向尾流）。"""
    return generate_grid_layout(
        domain,
        n_turbines,
        rotor_diameters,
        min_multiple=min_multiple,
        rng=rng,
        staggered=True,
    )
