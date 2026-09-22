"""场地边界约束。

支持任意（含凹陷）多边形边界，使用射线法判断点是否在多边形内。
``validate_polygon_geometry`` 提供严格的几何数据校验，租赁边界与禁建
多边形共用该校验。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


def signed_area(vertices: np.ndarray) -> float:
    """多边形有向面积（shoelace 公式），符号表示环绕方向。"""
    vertices = np.asarray(vertices, dtype=np.float64)
    x = vertices[:, 0]
    y = vertices[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _segments_intersect(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> bool:
    """判断线段 p1p2 与 p3p4 是否相交（含端点接触与共线重叠）。"""

    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a, b, c):
        return (
            min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9
            and min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9
            and abs(orient(a, b, c)) <= 1e-7
        )

    d1 = orient(p3, p4, p1)
    d2 = orient(p3, p4, p2)
    d3 = orient(p1, p2, p3)
    d4 = orient(p1, p2, p4)

    if ((d1 > 1e-7 and d2 < -1e-7) or (d1 < -1e-7 and d2 > 1e-7)) and (
        (d3 > 1e-7 and d4 < -1e-7) or (d3 < -1e-7 and d4 > 1e-7)
    ):
        return True

    if abs(d1) <= 1e-7 and on_segment(p3, p4, p1):
        return True
    if abs(d2) <= 1e-7 and on_segment(p3, p4, p2):
        return True
    if abs(d3) <= 1e-7 and on_segment(p1, p2, p3):
        return True
    if abs(d4) <= 1e-7 and on_segment(p1, p2, p4):
        return True
    return False


def validate_polygon_geometry(
    vertices: np.ndarray,
    name: str = "多边形",
    min_vertices: int = 3,
) -> np.ndarray:
    """严格校验多边形顶点数据。

    检查项：形状 (N,2)、数值有限、顶点数、连续重复点、退化（零面积）、
    非相邻边自相交。允许凹陷多边形，但边不允许互相穿越。

    Returns
    -------
    np.ndarray
        校验通过后的 float64 顶点数组。

    Raises
    ------
    ValueError
        任一校验失败，错误信息包含多边形名称与具体原因。
    """
    arr = np.asarray(vertices, dtype=np.float64)

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name}: 顶点坐标必须是形状为 (N, 2) 的数组，实际形状为 {arr.shape}")
    if arr.shape[0] < min_vertices:
        raise ValueError(f"{name}: 至少需要 {min_vertices} 个顶点，实际只有 {arr.shape[0]} 个")
    if not np.all(np.isfinite(arr)):
        bad = int(np.where(~np.isfinite(arr).all(axis=1))[0][0])
        raise ValueError(f"{name}: 第 {bad} 个顶点包含非有限数值 (NaN/Inf)")

    n = len(arr)

    for i in range(n):
        j = (i + 1) % n
        if np.linalg.norm(arr[j] - arr[i]) < 1e-9:
            raise ValueError(f"{name}: 第 {i} 与第 {j % n} 个顶点重合，存在退化边")

    if abs(signed_area(arr)) < 1.0:
        raise ValueError(f"{name}: 多边形面积接近零，可能退化或共线")

    for i in range(n):
        a1, a2 = arr[i], arr[(i + 1) % n]
        for j in range(i + 1, n):
            if (j + 1) % n == i or (i + 1) % n == j:
                continue  # 共享顶点的相邻边
            b1, b2 = arr[j], arr[(j + 1) % n]
            if _segments_intersect(a1, a2, b1, b2):
                raise ValueError(
                    f"{name}: 多边形自相交，边 ({i},{(i + 1) % n}) 与边 "
                    f"({j},{(j + 1) % n}) 相交"
                )

    return arr


@dataclass
class SiteBoundary:
    """场地边界类。

    使用闭合多边形定义场地范围，支持凹陷多边形。构造时执行严格几何校验。

    Parameters
    ----------
    vertices : np.ndarray
        多边形顶点坐标，形状为 (N, 2)，单位为米。
        多边形会自动闭合，不需要重复起点。
    name : str
        多边形名称，用于校验报错。
    """

    vertices: np.ndarray
    name: str = "租赁边界"

    def __post_init__(self) -> None:
        self.vertices = validate_polygon_geometry(self.vertices, self.name)

    @property
    def x_min(self) -> float:
        return float(np.min(self.vertices[:, 0]))

    @property
    def x_max(self) -> float:
        return float(np.max(self.vertices[:, 0]))

    @property
    def y_min(self) -> float:
        return float(np.min(self.vertices[:, 1]))

    @property
    def y_max(self) -> float:
        return float(np.max(self.vertices[:, 1]))

    @property
    def area(self) -> float:
        """多边形面积（shoelace 公式）。"""
        return abs(signed_area(self.vertices))

    def contains_point(
        self,
        point: np.ndarray,
        tolerance: float = 1e-9,
    ) -> bool:
        """判断点是否在多边形内部（射线法）。

        Parameters
        ----------
        point : np.ndarray
            点坐标，形状为 (2,)
        tolerance : float
            边界判定容差

        Returns
        -------
        bool
            True 表示点在多边形内部或边界上
        """
        pt = np.asarray(point, dtype=np.float64)
        verts = self.vertices

        if self._on_edge(pt, tolerance):
            return True

        n = len(verts)
        inside = False
        x, y = pt[0], pt[1]

        for i in range(n):
            j = (i + 1) % n
            xi, yi = verts[i]
            xj, yj = verts[j]

            if ((yi > y) != (yj > y)):
                x_intersect = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x <= x_intersect + tolerance:
                    inside = not inside

        return inside

    def _on_edge(self, point: np.ndarray, tolerance: float) -> bool:
        """检查点是否在多边形边界上。"""
        verts = self.vertices
        n = len(verts)

        for i in range(n):
            j = (i + 1) % n
            if self._point_on_segment(point, verts[i], verts[j], tolerance):
                return True
        return False

    @staticmethod
    def _point_on_segment(
        point: np.ndarray,
        seg_start: np.ndarray,
        seg_end: np.ndarray,
        tolerance: float,
    ) -> bool:
        """判断点是否在线段上。"""
        cross = (point[0] - seg_start[0]) * (seg_end[1] - seg_start[1]) - \
                (point[1] - seg_start[1]) * (seg_end[0] - seg_start[0])
        if abs(cross) > tolerance:
            return False

        dot = (point[0] - seg_start[0]) * (seg_end[0] - seg_start[0]) + \
              (point[1] - seg_start[1]) * (seg_end[1] - seg_start[1])
        if dot < -tolerance:
            return False

        len_sq = (seg_end[0] - seg_start[0]) ** 2 + (seg_end[1] - seg_start[1]) ** 2
        if dot > len_sq + tolerance:
            return False

        return True

    def contains_all(self, positions: np.ndarray) -> np.ndarray:
        """批量检查多个点是否在多边形内部。

        Parameters
        ----------
        positions : np.ndarray
            点坐标，形状为 (N, 2)

        Returns
        -------
        np.ndarray
            布尔数组，形状为 (N,)
        """
        positions = np.asarray(positions, dtype=np.float64)
        result = np.zeros(positions.shape[0], dtype=bool)
        for i, pt in enumerate(positions):
            result[i] = self.contains_point(pt)
        return result

    def project_to_boundary(self, point: np.ndarray) -> np.ndarray:
        """将点投影到多边形边界上（最近点）。

        Parameters
        ----------
        point : np.ndarray
            原始点坐标，形状为 (2,)

        Returns
        -------
        np.ndarray
            投影后的点坐标，形状为 (2,)
        """
        pt = np.asarray(point, dtype=np.float64)
        verts = self.vertices
        n = len(verts)

        best_dist = np.inf
        best_point = verts[0].copy()

        for i in range(n):
            j = (i + 1) % n
            proj = self._project_to_segment(pt, verts[i], verts[j])
            dist = np.linalg.norm(pt - proj)
            if dist < best_dist:
                best_dist = dist
                best_point = proj

        return best_point

    @staticmethod
    def _project_to_segment(
        point: np.ndarray,
        seg_start: np.ndarray,
        seg_end: np.ndarray,
    ) -> np.ndarray:
        """将点投影到线段上。"""
        seg_vec = seg_end - seg_start
        seg_len_sq = np.dot(seg_vec, seg_vec)

        if seg_len_sq < 1e-12:
            return seg_start.copy()

        t = np.dot(point - seg_start, seg_vec) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)

        return seg_start + t * seg_vec

    def sample_random_points(
        self,
        n_points: int,
        rng: Optional[np.random.Generator] = None,
        max_attempts: int = 100,
    ) -> np.ndarray:
        """在多边形内随机采样点（拒绝采样）。

        Parameters
        ----------
        n_points : int
            需要采样的点数
        rng : Optional[np.random.Generator]
            随机数生成器
        max_attempts : int
            每个点的最大尝试次数

        Returns
        -------
        np.ndarray
            采样点坐标，形状为 (n_points, 2)
        """
        if rng is None:
            rng = np.random.default_rng()

        points = np.zeros((n_points, 2), dtype=np.float64)
        x_min, x_max = self.x_min, self.x_max
        y_min, y_max = self.y_min, self.y_max

        for i in range(n_points):
            found = False
            for _ in range(max_attempts):
                x = rng.uniform(x_min, x_max)
                y = rng.uniform(y_min, y_max)
                pt = np.array([x, y])
                if self.contains_point(pt):
                    points[i] = pt
                    found = True
                    break
            if not found:
                raise RuntimeError(f"无法在场地内采样到第 {i+1} 个点")

        return points


def create_rectangular_boundary(
    width: float,
    height: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> SiteBoundary:
    """创建矩形场地边界。

    Parameters
    ----------
    width : float
        宽度（x方向）(m)
    height : float
        高度（y方向）(m)
    center_x : float
        中心x坐标 (m)
    center_y : float
        中心y坐标 (m)

    Returns
    -------
    SiteBoundary
        矩形场地边界
    """
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
    """创建正六边形场地边界。

    Parameters
    ----------
    radius : float
        外接圆半径 (m)
    center_x : float
        中心x坐标 (m)
    center_y : float
        中心y坐标 (m)

    Returns
    -------
    SiteBoundary
        六边形场地边界
    """
    angles = np.deg2rad(np.arange(0, 360, 60))
    vertices = np.column_stack([
        center_x + radius * np.cos(angles),
        center_y + radius * np.sin(angles),
    ])
    return SiteBoundary(vertices)


def create_irregular_boundary() -> SiteBoundary:
    """创建一个不规则多边形场地边界作为示例。

    Returns
    -------
    SiteBoundary
        不规则场地边界
    """
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
