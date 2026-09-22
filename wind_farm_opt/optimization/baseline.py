"""基线布局生成（规则网格）。

所有候选机位都必须落在可行域 :class:`Geofence` 内（租赁边界扣除带净距
的禁建区），并满足风机最小间距。无法在有限尝试内得到可行布局时抛出
:class:`InfeasibleLayoutError`，绝不返回违规布局。
"""

import numpy as np

from ..constraints.boundary import SiteBoundary
from ..constraints.geofence import Geofence, InfeasibleLayoutError
from ..constraints.spacing import (
    check_min_spacing,
    compute_min_spacing_from_diameters,
    enforce_min_spacing,
)


def _as_geofence(site: "SiteBoundary | Geofence") -> Geofence:
    """允许直接传入租赁边界（无禁建区）以保持向后兼容。"""
    if isinstance(site, Geofence):
        return site
    return Geofence(boundary=site)


def _grid_fill(
    site: Geofence,
    n_turbines: int,
    min_spacing: float,
    rng: np.random.Generator,
    grid_points: list[np.ndarray],
) -> np.ndarray:
    """以网格点起步，不足部分在可行域内随机补足，并执行间距修复。"""
    positions: list[np.ndarray] = []
    for pos in grid_points:
        if len(positions) >= n_turbines:
            break
        if not site.is_feasible_point(pos):
            continue
        if all(np.linalg.norm(pos - q) + 1e-9 >= min_spacing for q in positions):
            positions.append(pos)

    # 可行域内随机补足（拒绝采样本身带总尝试上限）
    if len(positions) < n_turbines:
        for _ in range(60):
            need = n_turbines - len(positions)
            try:
                candidates = site.sample_feasible_points(
                    need * 4, rng, max_attempts=4000
                )
            except InfeasibleLayoutError:
                break
            for cand in candidates:
                if len(positions) >= n_turbines:
                    break
                if all(np.linalg.norm(cand - q) + 1e-9 >= min_spacing for q in positions):
                    positions.append(cand)
            if len(positions) >= n_turbines:
                break

    if len(positions) < n_turbines:
        raise InfeasibleLayoutError(
            f"规则基线只能在可行域内容纳 {len(positions)}/{n_turbines} 台风机"
            f"（最小间距 {min_spacing:.0f} m），可行域可能被禁建区及其净距过度切割",
            violations=[],
        )

    positions_arr = np.array(positions, dtype=np.float64)

    valid, _ = check_min_spacing(positions_arr, min_spacing)
    feasible = site.feasible_mask(positions_arr).all()
    if not (valid and feasible):
        positions_arr = enforce_min_spacing(
            positions_arr, min_spacing, site, rng
        )

    return positions_arr


def generate_grid_layout(
    site: "SiteBoundary | Geofence",
    n_turbines: int,
    rotor_diameters: np.ndarray,
    min_multiple: float = 5.0,
    aspect_ratio: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """生成规则网格布局作为优化基线。

    Parameters
    ----------
    site : Geofence | SiteBoundary
        可行域（租赁边界 + 禁建区）；直接传 SiteBoundary 时视为无禁建区。
    n_turbines : int
        风机台数
    rotor_diameters : np.ndarray
        每台风机的转子直径
    min_multiple : float
        最小间距倍数
    aspect_ratio : float
        网格纵横比 (列数/行数)
    rng : Optional[np.random.Generator]
        随机数生成器

    Returns
    -------
    np.ndarray
        网格布局位置 (n_turbines, 2)

    Raises
    ------
    InfeasibleLayoutError
        可行域无法容纳指定数量的风机（含具体约束原因）。
    """
    if rng is None:
        rng = np.random.default_rng()

    site = _as_geofence(site)
    min_spacing = compute_min_spacing_from_diameters(rotor_diameters, min_multiple)

    n_rows = max(1, int(np.round(np.sqrt(n_turbines / aspect_ratio))))
    n_cols = max(1, int(np.ceil(n_turbines / n_rows)))

    x_min, x_max = site.x_min, site.x_max
    y_min, y_max = site.y_min, site.y_max

    margin = min_spacing * 0.5
    x_range = x_max - x_min - 2 * margin
    y_range = y_max - y_min - 2 * margin

    spacing_x = min(x_range / max(n_cols - 1, 1), min_spacing * 1.5)
    spacing_y = min(y_range / max(n_rows - 1, 1), min_spacing * 1.5)

    start_x = x_min + margin + (x_range - spacing_x * (n_cols - 1)) / 2.0
    start_y = y_min + margin + (y_range - spacing_y * (n_rows - 1)) / 2.0

    grid_points = [
        np.array([start_x + col * spacing_x, start_y + row * spacing_y])
        for row in range(n_rows)
        for col in range(n_cols)
    ]

    return _grid_fill(site, n_turbines, min_spacing, rng, grid_points)


def generate_staggered_grid_layout(
    site: "SiteBoundary | Geofence",
    n_turbines: int,
    rotor_diameters: np.ndarray,
    min_multiple: float = 5.0,
    dominant_direction: float = 270.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """生成交错网格布局（错位排列，减少主风向下的尾流）。

    Parameters
    ----------
    site : Geofence | SiteBoundary
        可行域（租赁边界 + 禁建区）。
    n_turbines : int
        风机台数
    rotor_diameters : np.ndarray
        每台风机的转子直径
    min_multiple : float
        最小间距倍数
    dominant_direction : float
        主风向（度），用于确定交错方向
    rng : Optional[np.random.Generator]
        随机数生成器

    Returns
    -------
    np.ndarray
        交错网格布局位置 (n_turbines, 2)
    """
    if rng is None:
        rng = np.random.default_rng()

    site = _as_geofence(site)
    min_spacing = compute_min_spacing_from_diameters(rotor_diameters, min_multiple)

    n_rows = max(1, int(np.sqrt(n_turbines)))
    n_cols = max(1, int(np.ceil(n_turbines / n_rows)))

    x_min, x_max = site.x_min, site.x_max
    y_min, y_max = site.y_min, site.y_max

    margin = min_spacing * 0.5
    x_range = x_max - x_min - 2 * margin
    y_range = y_max - y_min - 2 * margin

    spacing_x = max(x_range / max(n_cols - 1, 1), min_spacing * 1.2)
    spacing_y = max(y_range / max(n_rows - 1, 1), min_spacing * 1.2)

    start_x = x_min + margin + (x_range - spacing_x * (n_cols - 1)) / 2.0
    start_y = y_min + margin + (y_range - spacing_y * (n_rows - 1)) / 2.0

    grid_points = []
    for row in range(n_rows):
        offset = spacing_x / 2.0 if row % 2 == 1 else 0.0
        for col in range(n_cols):
            grid_points.append(
                np.array([start_x + col * spacing_x + offset, start_y + row * spacing_y])
            )

    return _grid_fill(site, n_turbines, min_spacing, rng, grid_points)
