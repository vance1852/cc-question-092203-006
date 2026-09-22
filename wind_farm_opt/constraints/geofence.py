"""禁建区与可行域（地理围栏）。

租赁边界内可能存在航道、海缆走廊、生态缓冲区等禁建多边形，且每个
禁建区有各自的安全净距（风轮外缘至禁建区边缘的距离）。本模块只依赖
numpy 实现：

- 严格的多边形几何校验（退化边、自相交、顶点数、有限数值等）；
- 有向/凹多边形的点包含、批量包含、点到多边形距离；
- ``Geofence`` 统一表达"租赁边界内、扣除所有带净距禁建区"的可行域；
- 有限时间内终止的拒绝采样、最近点投影与约束诊断。

约定：多边形顶点按顺序给出、首尾不重复；支持凹多边形，方向（顺/逆
时针）均可。风机以中心坐标参与计算，调用方需把"风轮外缘净距"换算为
中心净距（净距 + 风轮半径）后传入。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .boundary import SiteBoundary, signed_area, validate_polygon_geometry


# ---------------------------------------------------------------------------
# 禁建区
# ---------------------------------------------------------------------------

@dataclass
class ExclusionZone:
    """禁建多边形及其安全净距。

    Parameters
    ----------
    vertices : np.ndarray
        禁建区顶点 (N, 2)，可为凹多边形。
    setback : float
        风轮外缘至禁建区边缘的安全净距 (m)，必须为非负有限值。
    name : str
        禁建区名称（航道/海缆走廊/生态缓冲区…），用于诊断与图例。
    rotor_radius : float
        风轮半径 (m)。风机中心到禁建区的实际距离要求为
        ``setback + rotor_radius``。
    """

    vertices: np.ndarray
    setback: float = 0.0
    name: str = "禁建区"
    rotor_radius: float = 0.0

    def __post_init__(self) -> None:
        self.vertices = validate_polygon_geometry(np.asarray(self.vertices, dtype=np.float64), self.name)

        self.setback = float(self.setback)
        self.rotor_radius = float(self.rotor_radius)
        if not math.isfinite(self.setback) or self.setback < 0:
            raise ValueError(f"{self.name}: 净距必须为非负有限值，实际为 {self.setback}")
        if not math.isfinite(self.rotor_radius) or self.rotor_radius < 0:
            raise ValueError(f"{self.name}: 风轮半径必须为非负有限值，实际为 {self.rotor_radius}")

    @property
    def center_setback(self) -> float:
        """风机中心需保持的距离 = 净距 + 风轮半径。"""
        return self.setback + self.rotor_radius

    @property
    def area(self) -> float:
        return abs(signed_area(self.vertices))

    def contains_point(self, point: np.ndarray, tolerance: float = 1e-9) -> bool:
        """点是否落在禁建多边形内部或边界上（射线法，支持凹多边形）。"""
        pt = np.asarray(point, dtype=np.float64)
        verts = self.vertices
        n = len(verts)
        x, y = pt[0], pt[1]
        inside = False
        for i in range(n):
            xi, yi = verts[i]
            xj, yj = verts[(i + 1) % n]
            if (yi > y) != (yj > y):
                x_intersect = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x <= x_intersect + tolerance:
                    inside = not inside
        return inside

    def point_distance(self, point: np.ndarray) -> float:
        """点到禁建多边形的距离（内部点为 0，边界点为 0）。"""
        pt = np.asarray(point, dtype=np.float64)
        if self.contains_point(pt):
            return 0.0
        verts = self.vertices
        n = len(verts)
        best = np.inf
        for i in range(n):
            a = verts[i]
            b = verts[(i + 1) % n]
            seg = b - a
            length_sq = float(np.dot(seg, seg))
            t = 0.0 if length_sq < 1e-12 else float(np.clip(np.dot(pt - a, seg) / length_sq, 0.0, 1.0))
            proj = a + t * seg
            best = min(best, float(np.linalg.norm(pt - proj)))
        return best

    def violation_depth(self, point: np.ndarray) -> float:
        """点对该禁建区的违反深度 (m)。

        - 点在禁建区内（或边界上）：违反深度为所需中心净距；
        - 点在区外但距离小于所需净距：违反深度为净距差值；
        - 满足要求时为 0。
        """
        d = self.point_distance(point)
        required = self.center_setback
        if d + 1e-9 >= required:
            return 0.0
        if d <= 0.0:
            return required
        return required - d

    def nearest_point(self, point: np.ndarray) -> np.ndarray:
        """多边形边界上距给定点最近的点。"""
        pt = np.asarray(point, dtype=np.float64)
        verts = self.vertices
        n = len(verts)
        best_dist = np.inf
        best_pt = verts[0].copy()
        for i in range(n):
            a = verts[i]
            b = verts[(i + 1) % n]
            seg = b - a
            length_sq = float(np.dot(seg, seg))
            t = 0.0 if length_sq < 1e-12 else float(np.clip(np.dot(pt - a, seg) / length_sq, 0.0, 1.0))
            proj = a + t * seg
            d = float(np.linalg.norm(pt - proj))
            if d < best_dist:
                best_dist = d
                best_pt = proj
        return best_pt


# ---------------------------------------------------------------------------
# 可行域
# ---------------------------------------------------------------------------

@dataclass
class ConstraintViolation:
    """单台风机的一条约束违反诊断信息。"""

    turbine_idx: int
    constraint: str  # "lease_boundary" 或禁建区名称
    detail: str
    depth: float  # 违反深度 (m)


@dataclass
class Geofence:
    """租赁边界 + 若干带净距禁建区构成的可行域。

    风机位置可行当且仅当：

    1. 在租赁多边形内（可为凹多边形）；
    2. 对每个禁建区，风机中心到禁建多边形的距离
       ≥ ``setback + rotor_radius``。
    """

    boundary: SiteBoundary
    exclusions: list[ExclusionZone] = field(default_factory=list)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for k, zone in enumerate(self.exclusions):
            if zone.name in seen:
                raise ValueError(f"禁建区名称重复: {zone.name!r}")
            seen.add(zone.name)
            # 禁建区顶点必须落在租赁边界内（允许凹陷禁建区贴近边界）。
            for vi, v in enumerate(zone.vertices):
                if not self.boundary.contains_point(v):
                    raise ValueError(
                        f"{zone.name}: 第 {vi} 个顶点 {tuple(np.round(v, 1))} 位于租赁边界之外"
                    )

    # ----- 基本性质 -----

    @property
    def has_exclusions(self) -> bool:
        return len(self.exclusions) > 0

    @property
    def x_min(self) -> float:
        return self.boundary.x_min

    @property
    def x_max(self) -> float:
        return self.boundary.x_max

    @property
    def y_min(self) -> float:
        return self.boundary.y_min

    @property
    def y_max(self) -> float:
        return self.boundary.y_max

    @property
    def area(self) -> float:
        return self.boundary.area

    # ----- 可行性判定 -----

    def is_feasible_point(self, point: np.ndarray) -> bool:
        """单个候选点是否在可行域内。"""
        pt = np.asarray(point, dtype=np.float64)
        if not self.boundary.contains_point(pt):
            return False
        for zone in self.exclusions:
            if zone.violation_depth(pt) > 0:
                return False
        return True

    def feasible_mask(self, positions: np.ndarray) -> np.ndarray:
        """批量返回每台风机是否可行。"""
        positions = np.asarray(positions, dtype=np.float64)
        mask = np.ones(positions.shape[0], dtype=bool)
        for i, pt in enumerate(positions):
            mask[i] = self.is_feasible_point(pt)
        return mask

    # 兼容旧代码的别名
    def contains_point(self, point: np.ndarray) -> bool:
        return self.is_feasible_point(point)

    def contains_all(self, positions: np.ndarray) -> np.ndarray:
        return self.feasible_mask(positions)

    def diagnose(
        self,
        positions: np.ndarray,
        min_spacing: float = 0.0,
        spacing_matrix: Optional[np.ndarray] = None,
    ) -> list[ConstraintViolation]:
        """列出布局的全部约束违反（含具体禁建区名称与违反深度）。

        Parameters
        ----------
        positions : np.ndarray
            风机位置 (N, 2)。
        min_spacing : float
            最小风机间距 (m)，<=0 时跳过间距诊断。
        spacing_matrix : Optional[np.ndarray]
            可选的成对最小间距矩阵 (N, N)；缺省时使用标量 min_spacing。
        """
        positions = np.asarray(positions, dtype=np.float64)
        violations: list[ConstraintViolation] = []

        for i, pt in enumerate(positions):
            if not self.boundary.contains_point(pt):
                proj = self.boundary.project_to_boundary(pt)
                depth = float(np.linalg.norm(pt - proj))
                violations.append(
                    ConstraintViolation(
                        i,
                        "lease_boundary",
                        f"风机 #{i} 位于租赁边界外，距边界 {depth:.1f} m",
                        depth,
                    )
                )
                continue  # 出界点不再重复报告净距违反
            for zone in self.exclusions:
                depth = zone.violation_depth(pt)
                if depth > 0:
                    if zone.point_distance(pt) <= 0.0:
                        detail = (
                            f"风机 #{i} 位于{zone.name}内部，"
                            f"需外移至外缘净距≥{zone.setback:.0f} m（违反深度 {depth:.1f} m）"
                        )
                    else:
                        detail = (
                            f"风机 #{i} 距{zone.name}不足，"
                            f"要求外缘净距≥{zone.setback:.0f} m（违反深度 {depth:.1f} m）"
                        )
                    violations.append(ConstraintViolation(i, zone.name, detail, depth))

        if min_spacing > 0 or spacing_matrix is not None:
            n = len(positions)
            for i in range(n):
                for j in range(i + 1, n):
                    required = (
                        float(spacing_matrix[i, j])
                        if spacing_matrix is not None
                        else float(min_spacing)
                    )
                    dist = float(np.linalg.norm(positions[i] - positions[j]))
                    if dist + 1e-9 < required:
                        violations.append(
                            ConstraintViolation(
                                i,
                                "min_spacing",
                                f"风机 #{i} 与 #{j} 间距 {dist:.1f} m < 要求 {required:.1f} m",
                                required - dist,
                            )
                        )
        return violations

    def total_violation_depth(
        self, positions: np.ndarray, min_spacing: float = 0.0
    ) -> float:
        """所有约束违反深度之和 (m)，供惩罚函数使用。"""
        positions = np.asarray(positions, dtype=np.float64)
        total = 0.0
        for i, pt in enumerate(positions):
            if not self.boundary.contains_point(pt):
                proj = self.boundary.project_to_boundary(pt)
                total += float(np.linalg.norm(pt - proj))
                continue
            for zone in self.exclusions:
                total += zone.violation_depth(pt)
        if min_spacing > 0:
            n = len(positions)
            for i in range(n):
                for j in range(i + 1, n):
                    dist = float(np.linalg.norm(positions[i] - positions[j]))
                    if dist + 1e-9 < min_spacing:
                        total += min_spacing - dist
        return total

    # ----- 采样与投影 -----

    def sample_feasible_points(
        self,
        n_points: int,
        rng: np.random.Generator,
        max_attempts: int = 20000,
    ) -> np.ndarray:
        """在可行域内拒绝采样。

        按总尝试次数（而非每点次数）设上限，保证在密集/不可行场景下
        有限时间内终止。

        Raises
        ------
        InfeasibleLayoutError
            尝试次数耗尽仍无法采满，错误信息说明可行域命中率。
        """
        points = np.zeros((n_points, 2), dtype=np.float64)
        x_min, x_max = self.x_min, self.x_max
        y_min, y_max = self.y_min, self.y_max
        if x_max - x_min < 1e-9 or y_max - y_min < 1e-9:
            raise InfeasibleLayoutError(
                "租赁边界包围盒退化，无法在其中采样风机位置",
                violations=[],
            )

        filled = 0
        hits = 0
        attempts = 0
        batch = max(256, n_points * 4)
        while filled < n_points and attempts < max_attempts:
            remaining = min(batch, max_attempts - attempts)
            xs = rng.uniform(x_min, x_max, remaining)
            ys = rng.uniform(y_min, y_max, remaining)
            candidates = np.column_stack([xs, ys])
            for pt in candidates:
                attempts += 1
                if self.is_feasible_point(pt):
                    points[filled] = pt
                    filled += 1
                    hits += 1
                    if filled == n_points:
                        break
        if filled < n_points:
            hit_rate = hits / max(1, attempts) * 100.0
            raise InfeasibleLayoutError(
                f"可行域采样失败：{attempts} 次尝试仅命中 {hits} 个可行点"
                f"（命中率 {hit_rate:.2f}%），可行域可能过小或禁建区净距过大，"
                f"无法容纳 {n_points} 台风机",
                violations=[],
                hit_rate=hit_rate,
            )
        return points

    # 兼容旧名称
    def sample_random_points(
        self,
        n_points: int,
        rng: Optional[np.random.Generator] = None,
        max_attempts: int = 20000,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        return self.sample_feasible_points(n_points, rng, max_attempts)

    def project_into_feasible_region(
        self,
        point: np.ndarray,
        rng: np.random.Generator,
        max_attempts: int = 400,
        jitter: float = 20.0,
    ) -> np.ndarray:
        """把单个点投影回可行域。

        先投回租赁边界，再沿"远离最近禁建区"的方向逐步外推并配合抖动
        搜索满足全部净距的位置。

        Raises
        ------
        InfeasibleLayoutError
            有限尝试内找不到满足净距的位置。
        """
        pt = np.asarray(point, dtype=np.float64).copy()

        if not self.boundary.contains_point(pt):
            pt = self.boundary.project_to_boundary(pt)

        if self._satisfies_exclusions(pt):
            return pt

        # 沿远离禁建区方向做退避搜索
        cur = pt.copy()
        for scale in (1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 13.0):
            for _ in range(max_attempts // 7):
                push = np.zeros(2, dtype=np.float64)
                for zone in self.exclusions:
                    depth = zone.violation_depth(cur)
                    if depth > 0:
                        nearest = zone.nearest_point(cur)
                        if zone.point_distance(cur) <= 0.0:
                            # 点在禁建区内部：朝最近边界方向穿出禁建区
                            direction = nearest - cur
                        else:
                            # 点在区外但过近：沿背离禁建区方向退避
                            direction = cur - nearest
                        norm = np.linalg.norm(direction)
                        if norm < 1e-12:
                            # 恰好在边界上时方向退化，用随机方向试探
                            direction = rng.standard_normal(2)
                            norm = np.linalg.norm(direction)
                        push += direction / norm * (depth * scale + 1e-6)
                candidate = cur + push
                candidate += rng.uniform(-jitter, jitter, 2)
                if not self.boundary.contains_point(candidate):
                    candidate = self.boundary.project_to_boundary(candidate)
                if self._satisfies_exclusions(candidate):
                    return candidate
                cur = candidate

        zone_names = [z.name for z in self.exclusions if z.violation_depth(cur) > 0]
        raise InfeasibleLayoutError(
            f"无法将点 {np.round(pt, 1)} 投影到可行域，持续违反禁建区: {', '.join(zone_names)}",
            violations=[],
        )

    def _satisfies_exclusions(self, point: np.ndarray) -> bool:
        for zone in self.exclusions:
            if zone.violation_depth(point) > 0:
                return False
        return True

    # 兼容旧名称
    def project_to_boundary(self, point: np.ndarray) -> np.ndarray:
        return self.boundary.project_to_boundary(point)

    def summary(self) -> dict:
        """可行域结构化摘要，用于结果输出。"""
        exclusion_area = sum(z.area for z in self.exclusions)
        return {
            "lease_area_km2": float(self.boundary.area / 1e6),
            "n_exclusions": len(self.exclusions),
            "exclusions": [
                {
                    "name": z.name,
                    "setback_m": z.setback,
                    "center_setback_m": z.center_setback,
                    "polygon_area_km2": float(z.area / 1e6),
                    "n_vertices": int(len(z.vertices)),
                }
                for z in self.exclusions
            ],
            "exclusion_polygon_area_km2": float(exclusion_area / 1e6),
        }


class InfeasibleLayoutError(RuntimeError):
    """布局在给定约束下不可行（采样/修复在有限时间内失败）。

    Attributes
    ----------
    violations : list[ConstraintViolation]
        具体约束违反列表（若已有候选布局）。
    hit_rate : Optional[float]
        随机采样可行域命中率（仅采样失败时给出）。
    """

    def __init__(
        self,
        message: str,
        violations: Optional[list[ConstraintViolation]] = None,
        hit_rate: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.violations = violations or []
        self.hit_rate = hit_rate

    def violation_report(self) -> str:
        """生成带具体约束原因的多行失败报告。"""
        lines = [str(self)]
        if self.hit_rate is not None:
            lines.append(f"  可行域随机命中率: {self.hit_rate:.3f}%")
        if self.violations:
            lines.append(f"  共 {len(self.violations)} 条约束违反：")
            for v in self.violations:
                lines.append(f"    - [{v.constraint}] {v.detail}")
        return "\n".join(lines)
