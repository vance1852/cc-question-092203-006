"""禁建区与可行域模型。

最新勘测图在租赁边界内部新增了航道、海缆走廊和生态缓冲区等禁建区，
风机塔位（及风轮边缘）不得进入这些区域，且必须与各区域保持不同的
安全净距。本模块提供：

- :class:`ExclusionZone`：单个禁建多边形（允许凹陷）及其安全净距；
- :class:`FeasibleDomain`：租赁边界扣除所有禁建区（含净距缓冲）后的
  可行域，支持包含判定、违规度量、随机采样与投影修复；
- :class:`LayoutViolationReport`：对一个机位方案的完整约束校验报告；
- :class:`InfeasibleLayoutError`：带具体约束原因的失败异常。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Optional

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon, Point, MultiPolygon
from shapely.ops import triangulate, nearest_points
from shapely.validation import explain_validity

from .boundary import SiteBoundary, GeometryValidationError, _EPS


# 允许的禁建区类别（仅用于配置校验与图例标注）。
ZONE_KINDS = (
    "shipping_lane",      # 航道
    "cable_corridor",     # 海缆走廊
    "ecological_buffer",  # 生态缓冲区
    "other",
)

_ZONE_KIND_LABELS = {
    "shipping_lane": "航道",
    "cable_corridor": "海缆走廊",
    "ecological_buffer": "生态缓冲区",
    "other": "禁建区",
}


class InfeasibleLayoutError(RuntimeError):
    """方案无法满足约束。

    Attributes
    ----------
    reasons : list[str]
        人类可读的具体约束原因。
    details : dict
        结构化细节（已布置台数、净距、违规明细等）。
    """

    def __init__(self, message: str, reasons: Optional[list[str]] = None,
                 details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.reasons = reasons or []
        self.details = details or {}


@dataclass
class ExclusionZone:
    """禁建区。

    Parameters
    ----------
    vertices : array-like
        禁建多边形顶点 (N, 2)，单位米，允许凹陷。
    setback : float
        风轮边缘到该禁建区的安全净距 (m)，不同区域可以不同。
    name : str
        区域名称。
    kind : str
        区域类别，见 :data:`ZONE_KINDS`。
    """

    vertices: np.ndarray
    setback: float
    name: str = "禁建区"
    kind: str = "other"

    def __post_init__(self) -> None:
        verts = np.asarray(self.vertices, dtype=np.float64)
        verts = self._validate_vertices(verts)
        object.__setattr__(self, "vertices", verts)

        if self.kind not in ZONE_KINDS:
            raise GeometryValidationError(
                f"禁建区 {self.name!r} 的类别 {self.kind!r} 非法，"
                f"可选: {ZONE_KINDS}"
            )
        setback = float(self.setback)
        if not np.isfinite(setback) or setback < 0:
            raise GeometryValidationError(
                f"禁建区 {self.name!r} 的安全净距必须是非负有限值，得到 {self.setback}"
            )
        object.__setattr__(self, "setback", setback)

        polygon = ShapelyPolygon(verts)
        if not polygon.is_valid:
            raise GeometryValidationError(
                f"禁建区 {self.name!r} 不是有效多边形: {explain_validity(polygon)}"
            )
        object.__setattr__(self, "_polygon", polygon)

    @staticmethod
    def _validate_vertices(verts: np.ndarray) -> np.ndarray:
        if verts.ndim != 2 or verts.shape[1] != 2:
            raise GeometryValidationError("禁建区顶点必须是形状为 (N, 2) 的数组")
        if not np.isfinite(verts).all():
            raise GeometryValidationError("禁建区顶点包含 NaN 或无穷大")
        if verts.shape[0] < 3:
            raise GeometryValidationError("禁建多边形至少需要3个顶点")

        if np.allclose(verts[0], verts[-1], atol=_EPS, rtol=0.0):
            verts = verts[:-1]
        if verts.shape[0] < 3:
            raise GeometryValidationError("禁建多边形至少需要3个不重复顶点")

        n = verts.shape[0]
        for i in range(n):
            j = (i + 1) % n
            if np.linalg.norm(verts[j] - verts[i]) < _EPS:
                raise GeometryValidationError(
                    f"禁建多边形存在重合相邻顶点: 顶点 {i} 与 {j}"
                )

        area2 = 0.0
        for i in range(n):
            j = (i + 1) % n
            area2 += verts[i, 0] * verts[j, 1] - verts[j, 0] * verts[i, 1]
        if abs(area2) < _EPS:
            raise GeometryValidationError(
                "禁建多边形有向面积为零（顶点可能共线，或自相交形成正负抵消）"
            )
        return verts

    @property
    def polygon(self) -> ShapelyPolygon:
        return self._polygon

    @property
    def area(self) -> float:
        return float(self._polygon.area)

    @property
    def kind_label(self) -> str:
        return _ZONE_KIND_LABELS[self.kind]

    def buffered(self, extra_distance: float = 0.0) -> ShapelyPolygon:
        """多边形外扩（安全净距 + 附加距离，如风轮半径）后的区域。"""
        return self._polygon.buffer(self.setback + float(extra_distance),
                                    join_style=2)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "kind_label": self.kind_label,
            "setback_m": self.setback,
            "area_km2": self.area / 1e6,
            "vertices": self.vertices.tolist(),
        }


@dataclass
class LayoutViolationReport:
    """机位方案的约束校验结果。"""

    n_turbines: int
    feasible: bool
    out_of_lease: list[dict] = field(default_factory=list)
    zone_violations: list[dict] = field(default_factory=list)
    spacing_violations: list[dict] = field(default_factory=list)

    @property
    def reasons(self) -> list[str]:
        """生成具体、可操作的中文约束原因列表。"""
        msg: list[str] = []
        if self.out_of_lease:
            idxs = ", ".join(f"#{v['turbine']}" for v in self.out_of_lease[:10])
            req = self.out_of_lease[0].get("required_m")
            extra = (
                f"（风轮边缘距租赁边线需≥{req:.0f} m）" if req else ""
            )
            msg.append(
                f"{len(self.out_of_lease)} 台风机塔位越出租赁安装范围"
                f"{extra}（{idxs}{' …' if len(self.out_of_lease) > 10 else ''}）"
            )
        if self.zone_violations:
            by_zone: dict[str, int] = {}
            for v in self.zone_violations:
                by_zone[v["zone"]] = by_zone.get(v["zone"], 0) + 1
            for zname, count in by_zone.items():
                worst = max(
                    (v for v in self.zone_violations if v["zone"] == zname),
                    key=lambda v: v["shortfall_m"],
                )
                msg.append(
                    f"{count} 台风机侵入禁建区 {zname!r} 的安全净距 "
                    f"（要求风轮边缘净距≥{worst['required_m']:.0f} m，"
                    f"最近仅 {worst['actual_m']:.0f} m，"
                    f"欠 {worst['shortfall_m']:.0f} m）"
                )
        if self.spacing_violations:
            worst = max(self.spacing_violations, key=lambda v: v["shortfall_m"])
            msg.append(
                f"{len(self.spacing_violations)} 对风机间距不足 "
                f"（最小间距 {worst['required_m']:.0f} m，"
                f"最紧的一对 #{worst['turbine_i']}/#{worst['turbine_j']} "
                f"仅 {worst['actual_m']:.0f} m）"
            )
        return msg

    def to_dict(self) -> dict:
        return {
            "feasible": self.feasible,
            "reasons": self.reasons,
            "n_out_of_lease": len(self.out_of_lease),
            "n_zone_violations": len(self.zone_violations),
            "n_spacing_violations": len(self.spacing_violations),
            "out_of_lease": self.out_of_lease,
            "zone_violations": self.zone_violations,
            "spacing_violations": self.spacing_violations,
        }


class FeasibleDomain:
    """可行域：租赁边界扣除全部禁建区（含安全净距与风轮半径）。

    风机判定以塔位为点，但缓冲距离取 ``setback + rotor_radius``，
    从而保证“风轮边缘”与禁建区边缘保持配置的净距。
    """

    def __init__(
        self,
        lease: SiteBoundary,
        zones: Optional[list[ExclusionZone]] = None,
        rotor_radius: float = 0.0,
        lease_setback: float = 0.0,
    ) -> None:
        self.lease = lease
        self.zones = list(zones or [])
        if rotor_radius < 0 or not np.isfinite(rotor_radius):
            raise GeometryValidationError("风轮半径必须是非负有限值")
        if lease_setback < 0 or not np.isfinite(lease_setback):
            raise GeometryValidationError("租赁边界净距必须是非负有限值")
        self.rotor_radius = float(rotor_radius)
        self.lease_setback = float(lease_setback)

        names = [z.name for z in self.zones]
        if len(names) != len(set(names)):
            raise GeometryValidationError("禁建区名称必须唯一")

        base = lease.polygon
        # 塔位需内缩“边界净距 + 风轮半径”，才能保证风轮边缘距租赁
        # 边线至少 lease_setback（与禁建区的净距口径一致）。
        total_lease_inset = self.lease_setback + self.rotor_radius
        if total_lease_inset > 0:
            base = base.buffer(-total_lease_inset, join_style=2)
            if base.is_empty:
                raise GeometryValidationError(
                    f"租赁边界内缩 {total_lease_inset:.0f} m（边界净距 "
                    f"{self.lease_setback:.0f} + 风轮半径 {self.rotor_radius:.0f}）"
                    "后没有可安装区域"
                )

        forbidden = []
        for zone in self.zones:
            buf = zone.buffered(self.rotor_radius)
            if not buf.intersects(lease.polygon):
                raise GeometryValidationError(
                    f"禁建区 {zone.name!r}（含安全净距）与租赁边界完全不相交，"
                    "请检查坐标配置"
                )
            forbidden.append(buf.intersection(lease.polygon))

        feasible = base
        for buf in forbidden:
            feasible = feasible.difference(buf)
        feasible = feasible.buffer(0)

        if feasible.is_empty or feasible.area <= _EPS:
            zone_desc = "、".join(
                f"{z.name}（净距{z.setback:.0f} m）" for z in self.zones
            )
            raise InfeasibleLayoutError(
                "禁建区扣除后租赁场内没有任何可安装区域",
                reasons=[f"以下禁建区覆盖了全部租赁范围: {zone_desc}"],
                details={
                    "reason": "empty_feasible_domain",
                    "lease_area_km2": lease.area / 1e6,
                    "zones": [z.to_dict() for z in self.zones],
                },
            )

        self._base = base
        #: 塔位必须落入的“内缩后租赁区”（已含边界净距与风轮半径）。
        self._tower_lease = base
        self._forbidden = forbidden
        if feasible.geom_type == "Polygon":
            self._geometry = MultiPolygon([feasible])
        elif feasible.geom_type == "MultiPolygon":
            self._geometry = feasible
        else:  # GeometryCollection（含零散线/面），只保留面
            parts = [g for g in getattr(feasible, "geoms", [])
                     if g.geom_type == "Polygon" and not g.is_empty]
            if not parts:
                raise InfeasibleLayoutError(
                    "禁建区扣除后租赁场内没有任何可安装区域",
                    reasons=["可行域退化为线或点"]
                )
            self._geometry = MultiPolygon(parts)

    # ---- 基本属性 -------------------------------------------------
    @property
    def area(self) -> float:
        return float(self._geometry.area)

    @property
    def polygon(self):
        """底层 shapely 可行域几何（MultiPolygon）。"""
        return self._geometry

    @property
    def parts(self) -> list[ShapelyPolygon]:
        return list(self._geometry.geoms)

    @property
    def x_min(self) -> float:
        return float(self.lease.x_min)

    @property
    def x_max(self) -> float:
        return float(self.lease.x_max)

    @property
    def y_min(self) -> float:
        return float(self.lease.y_min)

    @property
    def y_max(self) -> float:
        return float(self.lease.y_max)

    # ---- 包含判定 -------------------------------------------------
    def contains_point(self, point: np.ndarray, tolerance: float = 1e-9) -> bool:
        """塔位点是否位于可行域内（含边界）。"""
        pt = Point(np.asarray(point, dtype=np.float64))
        return bool(
            self._geometry.covers(pt)
            or self._geometry.distance(pt) <= tolerance
        )

    def contains_all(self, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.float64)
        return np.array([self.contains_point(p) for p in positions], dtype=bool)

    def project_into(self, point: np.ndarray) -> np.ndarray:
        """把域外的点投影回可行域上最近的位置。"""
        pt = Point(np.asarray(point, dtype=np.float64))
        if self._geometry.covers(pt):
            return np.asarray(point, dtype=np.float64)
        # nearest_points(domain, point) 返回
        # (domain 上最近点, point 上最近点即其自身)，取第一个。
        nearest_on_domain, _ = nearest_points(self._geometry, pt)
        return np.array([nearest_on_domain.x, nearest_on_domain.y],
                        dtype=np.float64)

    # ---- 违规度量 -------------------------------------------------
    def point_zone_shortfall(self, point: np.ndarray) -> list[tuple[ExclusionZone, float, float, float]]:
        """返回该点对各禁建区的净距欠账。

        Returns
        -------
        list of (zone, required, actual, shortfall)
            required = 净距 + 风轮半径；actual 为塔位到禁建多边形的
            距离（在区内为 0）；shortfall = required - actual。
            仅返回 shortfall > 0 的项。
        """
        pt = Point(np.asarray(point, dtype=np.float64))
        out = []
        for zone in self.zones:
            required = zone.setback + self.rotor_radius
            actual = float(zone.polygon.distance(pt))
            shortfall = required - actual
            if shortfall > 1e-9:
                out.append((zone, required, actual, shortfall))
        return out

    def validate_layout(
        self,
        positions: np.ndarray,
        min_spacing: Optional[float] = None,
    ) -> LayoutViolationReport:
        """全面校验一个机位方案（租赁边界 / 禁建净距 / 风机间距）。"""
        positions = np.asarray(positions, dtype=np.float64)
        n = positions.shape[0]
        report = LayoutViolationReport(n_turbines=n, feasible=True)

        for i, pos in enumerate(positions):
            if not np.isfinite(pos).all():
                report.feasible = False
                report.out_of_lease.append({
                    "turbine": i, "actual_m": float("inf"),
                })
                continue
            pt = Point(pos)
            # 塔位必须位于内缩后的租赁区内（含边界净距与风轮半径）。
            if not self._tower_lease.covers(pt):
                report.out_of_lease.append({
                    "turbine": i,
                    "required_m": float(self.lease_setback + self.rotor_radius),
                    "actual_m": float(self.lease.polygon.distance(pt)),
                    "shortfall_m": float(self._tower_lease.distance(pt)),
                })
            for zone, required, actual, shortfall in self.point_zone_shortfall(pos):
                report.zone_violations.append({
                    "turbine": i,
                    "zone": zone.name,
                    "zone_kind": zone.kind,
                    "required_m": float(required),
                    "actual_m": float(actual),
                    "shortfall_m": float(shortfall),
                })

        if min_spacing is not None:
            for i in range(n):
                for j in range(i + 1, n):
                    dist = float(np.linalg.norm(positions[i] - positions[j]))
                    if dist < min_spacing - 1e-9:
                        report.spacing_violations.append({
                            "turbine_i": i,
                            "turbine_j": j,
                            "required_m": float(min_spacing),
                            "actual_m": dist,
                            "shortfall_m": float(min_spacing - dist),
                        })

        report.feasible = not (
            report.out_of_lease
            or report.zone_violations
            or report.spacing_violations
        )
        return report

    def penalty(
        self,
        positions: np.ndarray,
        min_spacing: Optional[float] = None,
        factor: float = 1e6,
    ) -> tuple[float, dict]:
        """计算约束违反惩罚（GA/PSO 使用）。

        惩罚按“越界距离/净距欠账/间距欠账”的米数加权，使算法具有
        连续的梯度方向，而不仅是越界计数。返回 (penalty, metrics)。
        """
        positions = np.asarray(positions, dtype=np.float64)
        n = positions.shape[0]
        out_m = 0.0
        zone_m = 0.0
        n_zone_hits = 0
        for pos in positions:
            if not np.isfinite(pos).all():
                out_m += 1e6
                continue
            pt = Point(pos)
            d_lease = float(self._tower_lease.distance(pt))
            if not self._tower_lease.covers(pt):
                out_m += d_lease
            for _, _, _, shortfall in self.point_zone_shortfall(pos):
                zone_m += shortfall
                n_zone_hits += 1

        spacing_m = 0.0
        n_spacing_pairs = 0
        if min_spacing is not None:
            for i in range(n):
                for j in range(i + 1, n):
                    dist = float(np.linalg.norm(positions[i] - positions[j]))
                    if dist < min_spacing:
                        spacing_m += min_spacing - dist
                        n_spacing_pairs += 1

        penalty = factor * (out_m + zone_m + spacing_m)
        return penalty, {
            "out_of_lease_m": out_m,
            "zone_shortfall_m": zone_m,
            "n_zone_violations": n_zone_hits,
            "spacing_shortfall_m": spacing_m,
            "n_spacing_violations": n_spacing_pairs,
        }

    # ---- 随机采样 -------------------------------------------------
    def _build_sampling_triangles(self):
        """对可行域做受约束 Delaunay 三角剖分，用于按面积均匀采样。"""
        if hasattr(self, "_sampling_triangles"):
            return self._sampling_triangles

        triangles = []
        for part in self.parts:
            candidates = triangulate(part)
            for tri in candidates:
                clipped = tri.intersection(part)
                if clipped.is_empty or clipped.area <= _EPS:
                    continue
                if clipped.geom_type == "Polygon":
                    triangles.append(clipped)
                else:
                    triangles.extend(
                        g for g in clipped.geoms
                        if g.geom_type == "Polygon" and not g.is_empty
                    )
        areas = np.array([t.area for t in triangles], dtype=np.float64)
        self._sampling_triangles = triangles
        self._sampling_areas = areas
        self._sampling_cumarea = np.cumsum(areas / areas.sum())
        return triangles

    def _sample_uniform_point(
        self, rng: np.random.Generator
    ) -> np.ndarray:
        """在可行域内按面积均匀采样一个点。"""
        self._build_sampling_triangles()
        idx = int(np.searchsorted(self._sampling_cumarea, rng.random()))
        idx = min(idx, len(self._sampling_triangles) - 1)
        tri = self._sampling_triangles[idx]
        coords = np.asarray(tri.exterior.coords[:3], dtype=np.float64)
        r1, r2 = rng.random(), rng.random()
        if r1 + r2 > 1.0:
            r1, r2 = 1.0 - r1, 1.0 - r2
        point = coords[0] + r1 * (coords[1] - coords[0]) + r2 * (coords[2] - coords[0])
        return point

    def sample_points(
        self,
        n_points: int,
        rng: Optional[np.random.Generator] = None,
        max_attempts: int = 10000,
    ) -> np.ndarray:
        """在可行域内（不含任何禁建区及其净距）采样 n_points 个点。"""
        if rng is None:
            rng = np.random.default_rng()

        points = np.zeros((n_points, 2), dtype=np.float64)
        for i in range(n_points):
            for _ in range(max_attempts):
                pt = self._sample_uniform_point(rng)
                if self.contains_point(pt):
                    points[i] = pt
                    break
            else:
                raise InfeasibleLayoutError(
                    "可行域随机采样持续失败",
                    reasons=[
                        f"在采样第 {i + 1}/{n_points} 个点时，"
                        f"连续 {max_attempts} 次尝试未落入可行域",
                        "可行域可能过于破碎或狭窄",
                    ],
                    details={"reason": "sampling_failed", "placed": i},
                )
        return points

    def sample_separated_points(
        self,
        n_points: int,
        min_spacing: float,
        rng: Optional[np.random.Generator] = None,
        existing: Optional[np.ndarray] = None,
        candidate_budget: int = 20000,
        time_budget_s: float = 15.0,
    ) -> np.ndarray:
        """在可行域内采样满足最小两两间距的 n 个点。

        Parameters
        ----------
        n_points : int
            还需采样的点数
        min_spacing : float
            最小两两间距 (m)
        existing : np.ndarray, optional
            已存在的机位 (M, 2)，新采样点也要与之保持间距。

        采用顺序拒绝采样，受尝试次数与时间双重预算约束；超时则抛出
        :class:`InfeasibleLayoutError`，而不是无限重试。
        """
        if rng is None:
            rng = np.random.default_rng()
        start = time.monotonic()

        positions: list[np.ndarray] = []
        if existing is not None and len(existing):
            positions = [np.asarray(p, dtype=np.float64) for p in existing]
        n_existing = len(positions)
        attempts = 0
        while len(positions) - n_existing < n_points:
            if attempts >= candidate_budget:
                raise InfeasibleLayoutError(
                    f"无法在可行域内容纳 {n_points} 台保持 "
                    f"{min_spacing:.0f} m 间距的风机",
                    reasons=self._packing_failure_reasons(
                        n_points, len(positions), min_spacing, attempts
                    ),
                    details={
                        "reason": "packing_failed",
                        "required": n_points,
                        "placed": len(positions),
                        "min_spacing_m": float(min_spacing),
                        "attempts": attempts,
                        "feasible_area_km2": self.area / 1e6,
                    },
                )
            if time.monotonic() - start > time_budget_s:
                raise InfeasibleLayoutError(
                    f"可行域内布置 {n_points} 台风机的采样超过 "
                    f"{time_budget_s:.0f}s 时间预算",
                    reasons=self._packing_failure_reasons(
                        n_points, len(positions), min_spacing, attempts
                    ),
                    details={
                        "reason": "time_budget_exceeded",
                        "required": n_points,
                        "placed": len(positions),
                        "min_spacing_m": float(min_spacing),
                        "feasible_area_km2": self.area / 1e6,
                    },
                )

            candidate = self._sample_uniform_point(rng)
            attempts += 1
            if all(np.linalg.norm(candidate - p) >= min_spacing
                   for p in positions):
                positions.append(candidate)

        total = np.array(positions, dtype=np.float64)
        return total[n_existing:] if n_existing else total

    def _packing_failure_reasons(
        self, required: int, placed: int, min_spacing: float, attempts: int
    ) -> list[str]:
        reasons = [
            f"仅成功布置 {placed}/{required} 台风机（最小间距 "
            f"{min_spacing:.0f} m，已尝试 {attempts} 个候选点）"
        ]
        reasons.append(
            f"扣除禁建区与安全净距后的可行域面积为 {self.area / 1e6:.3f} km²，"
            f"而租赁边界总面积为 {self.lease.area / 1e6:.3f} km²"
        )
        for zone in self.zones:
            reasons.append(
                f"禁建区 {zone.name!r}（{zone.kind_label}）："
                f"面积 {zone.area / 1e6:.3f} km²，安全净距 {zone.setback:.0f} m"
            )
        return reasons

    # ---- 修复 -----------------------------------------------------
    def repair_layout(
        self,
        positions: np.ndarray,
        min_spacing: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
        max_iterations: int = 300,
        time_budget_s: float = 10.0,
    ) -> np.ndarray:
        """通过推开碰撞对 + 投影回可行域来修复布局。

        有限次迭代/限时内无法完全修复时抛出
        :class:`InfeasibleLayoutError`，绝不返回带违规的布局。
        """
        if rng is None:
            rng = np.random.default_rng()
        start = time.monotonic()

        positions = np.asarray(positions, dtype=np.float64).copy()
        n = positions.shape[0]

        # 先把所有点收回可行域。
        for k in range(n):
            positions[k] = self.project_into(positions[k])

        for _ in range(max_iterations):
            if time.monotonic() - start > time_budget_s:
                break

            report = self.validate_layout(positions, min_spacing)
            if report.feasible:
                return positions

            for viol in report.spacing_violations:
                i, j = viol["turbine_i"], viol["turbine_j"]
                vec = positions[j] - positions[i]
                dist = float(np.linalg.norm(vec))
                if dist < 1e-9:
                    vec = rng.standard_normal(2)
                    dist = float(np.linalg.norm(vec))
                push = (min_spacing - dist) / 2.0 + 1e-6
                unit = vec / dist
                positions[i] -= unit * push
                positions[j] += unit * push

            for k in range(n):
                positions[k] = self.project_into(positions[k])

        report = self.validate_layout(positions, min_spacing)
        if report.feasible:
            return positions

        raise InfeasibleLayoutError(
            "间距/禁建修复在有限迭代内无法得到可行布局",
            reasons=report.reasons,
            details={
                "reason": "repair_failed",
                "report": report.to_dict(),
                "max_iterations": max_iterations,
            },
        )
