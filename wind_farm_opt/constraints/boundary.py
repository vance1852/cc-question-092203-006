"""场地租赁边界。

支持任意（可凹陷）多边形边界，基于 shapely 做健壮的点包含、投影、
面积计算，并提供严格的几何数据校验。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point
from shapely.validation import explain_validity


class GeometryValidationError(ValueError):
    """几何数据校验失败。"""


# 判定重复点/共线拼接时允许的坐标容差（米）。场地坐标量级为千米，
# 1e-6 m 的容差足以识别数值噪声又不会误并真实顶点。
_EPS = 1e-6


@dataclass
class SiteBoundary:
    """场地租赁边界类。

    使用闭合多边形定义租赁范围（允许凹陷）。内部持有一个 shapely
    多边形用于几何运算，并暴露射线法风格的接口供其它模块使用。

    Parameters
    ----------
    vertices : array-like
        多边形顶点坐标，形状为 (N, 2)，单位为米。
        多边形会自动闭合，不需要重复起点。
    name : str
        边界名称（用于图表与结果标注）。
    """

    vertices: np.ndarray
    name: str = "租赁边界"

    def __post_init__(self) -> None:
        verts = self._validate_vertices(np.asarray(self.vertices, dtype=np.float64))
        object.__setattr__(self, "vertices", verts)
        polygon = ShapelyPolygon(verts)
        if not polygon.is_valid:
            reason = explain_validity(polygon)
            raise GeometryValidationError(f"租赁边界不是有效多边形: {reason}")
        object.__setattr__(self, "_polygon", polygon)
        object.__setattr__(self, "_exterior", polygon.exterior)

    @staticmethod
    def _validate_vertices(verts: np.ndarray) -> np.ndarray:
        """严格校验顶点数据，返回去除闭合重复点后的顶点数组。"""
        if verts.ndim != 2 or verts.shape[1] != 2:
            raise GeometryValidationError("顶点坐标必须是形状为 (N, 2) 的数组")
        if not np.isfinite(verts).all():
            raise GeometryValidationError("顶点坐标包含 NaN 或无穷大")
        if verts.shape[0] < 3:
            raise GeometryValidationError("多边形至少需要3个不重复顶点")

        # 去掉用户可能重复给出的闭合点（最后一个点等于第一个点）。
        if np.allclose(verts[0], verts[-1], atol=_EPS, rtol=0.0):
            verts = verts[:-1]
        if verts.shape[0] < 3:
            raise GeometryValidationError("多边形至少需要3个不重复顶点")

        # 检查相邻重复点（含首尾），否则会产生退化边。
        n = verts.shape[0]
        for i in range(n):
            j = (i + 1) % n
            if np.linalg.norm(verts[j] - verts[i]) < _EPS:
                raise GeometryValidationError(
                    f"多边形存在重合相邻顶点: 顶点 {i} 与 {j} 坐标几乎相同"
                )

        area2 = 0.0
        for i in range(n):
            j = (i + 1) % n
            area2 += verts[i, 0] * verts[j, 1] - verts[j, 0] * verts[i, 1]
        if abs(area2) < _EPS:
            raise GeometryValidationError(
                "多边形有向面积为零（顶点可能共线，或自相交形成正负抵消）"
            )

        return verts

    @property
    def polygon(self) -> ShapelyPolygon:
        """底层 shapely 多边形。"""
        return self._polygon

    @property
    def x_min(self) -> float:
        return float(self.vertices[:, 0].min())

    @property
    def x_max(self) -> float:
        return float(self.vertices[:, 0].max())

    @property
    def y_min(self) -> float:
        return float(self.vertices[:, 1].min())

    @property
    def y_max(self) -> float:
        return float(self.vertices[:, 1].max())

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return tuple(float(v) for v in self._polygon.bounds)

    @property
    def area(self) -> float:
        """多边形面积（shoelace 与 shapely 一致）。"""
        return float(self._polygon.area)

    def contains_point(self, point: np.ndarray, tolerance: float = 1e-9) -> bool:
        """判断塔位点是否在租赁边界内（含边界）。"""
        pt = Point(np.asarray(point, dtype=np.float64))
        return bool(
            self._polygon.covers(pt)
            or self._polygon.distance(pt) <= tolerance
        )

    def contains_all(self, positions: np.ndarray) -> np.ndarray:
        """批量检查点是否在租赁边界内，返回形状 (N,) 的布尔数组。"""
        positions = np.asarray(positions, dtype=np.float64)
        return np.array([self.contains_point(p) for p in positions], dtype=bool)

    def project_to_boundary(
        self, point: np.ndarray, inward: bool = True
    ) -> np.ndarray:
        """返回边界上距该点最近的点；点在外则自动落在边界上。"""
        pt = Point(np.asarray(point, dtype=np.float64))
        nearest = self._exterior.interpolate(self._exterior.project(pt))
        return np.array([nearest.x, nearest.y], dtype=np.float64)

    def sample_random_points(
        self,
        n_points: int,
        rng: Optional[np.random.Generator] = None,
        max_attempts: int = 100,
    ) -> np.ndarray:
        """在租赁边界内（不含禁建区——该方法不感知禁建区，仅保留兼容）。

        禁建区感知的采样请使用
        :meth:`wind_farm_opt.constraints.site.FeasibleDomain.sample_points`。
        """
        if rng is None:
            rng = np.random.default_rng()

        points = np.zeros((n_points, 2), dtype=np.float64)
        x_min, y_min, x_max, y_max = self.bounds

        for i in range(n_points):
            found = False
            for _ in range(max_attempts):
                pt = np.array([
                    rng.uniform(x_min, x_max),
                    rng.uniform(y_min, y_max),
                ])
                if self.contains_point(pt):
                    points[i] = pt
                    found = True
                    break
            if not found:
                raise RuntimeError(f"无法在租赁边界内采样到第 {i + 1} 个点")
        return points


def create_rectangular_boundary(
    width: float,
    height: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> SiteBoundary:
    """创建矩形租赁边界。"""
    if width <= 0 or height <= 0:
        raise GeometryValidationError("矩形边界的宽度和高度必须为正数")
    x1 = center_x - width / 2.0
    x2 = center_x + width / 2.0
    y1 = center_y - height / 2.0
    y2 = center_y + height / 2.0
    vertices = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
    ], dtype=np.float64)
    return SiteBoundary(vertices)


def create_hexagonal_boundary(
    radius: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> SiteBoundary:
    """创建正六边形租赁边界。"""
    if radius <= 0:
        raise GeometryValidationError("六边形外接圆半径必须为正数")
    angles = np.deg2rad(np.arange(0, 360, 60))
    vertices = np.column_stack([
        center_x + radius * np.cos(angles),
        center_y + radius * np.sin(angles),
    ])
    return SiteBoundary(vertices)


def create_irregular_boundary() -> SiteBoundary:
    """创建一个不规则（凹陷）多边形租赁边界作为示例。"""
    vertices = np.array([
        [0.0, 0.0],
        [3000.0, -200.0],
        [3200.0, 1500.0],
        [2800.0, 2800.0],
        [1500.0, 3000.0],
        [-200.0, 2500.0],
        [-300.0, 1200.0],
    ], dtype=np.float64)
    return SiteBoundary(vertices)
