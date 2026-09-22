"""配置管理模块。

用于从JSON/YAML文件加载配置，或通过命令行参数构建配置。
"""

import json
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np

from .core.turbine import Turbine, create_default_turbine
from .core.wind_resource import WindResource, create_default_wind_resource
from .core.wake import JensenWake, GaussianWake, WakeModel
from .constraints.boundary import (
    SiteBoundary,
    create_rectangular_boundary,
    create_hexagonal_boundary,
    create_irregular_boundary,
)
from .constraints.geofence import Geofence, ExclusionZone


@dataclass
class OptimizationConfig:
    """优化算法配置。"""
    algorithm: str = "ga"
    population_size: int = 40
    max_iterations: int = 80
    min_spacing_multiple: float = 5.0
    seed: Optional[int] = 42


@dataclass
class VisualizationConfig:
    """可视化配置。"""
    save_dir: str = "output"
    save_plots: bool = True
    show_plots: bool = False
    plot_wake_heatmap: bool = True


@dataclass
class EconomicConfig:
    """经济性分析配置。"""
    electricity_price: float = 0.45
    discount_rate: float = 0.06
    enable_analysis: bool = True


@dataclass
class WindFarmConfig:
    """完整的风电场分析配置。"""
    n_turbines: int = 15
    turbine_model: str = "V126-3.45MW"
    wake_model: str = "jensen"
    wake_decay: float = 0.07
    superposition_method: str = "sum_of_squares"

    boundary_type: str = "rectangular"
    boundary_params: dict = field(default_factory=lambda: {
        "width": 4000,
        "height": 4000,
        "center_x": 0,
        "center_y": 0,
    })

    #: 禁建区列表，每项形如
    #: ``{"name": "航道", "setback": 200.0, "vertices": [[x, y], ...]}``。
    #: 顶点可为凹陷多边形，setback 为风轮外缘至禁建区边缘的净距 (m)。
    exclusions: list[dict] = field(default_factory=list)

    wind_resource_type: str = "default"
    wind_resource_params: dict = field(default_factory=lambda: {
        "num_sectors": 12,
        "dominant_direction": 270.0,
        "mean_speed": 8.5,
    })

    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    economic: EconomicConfig = field(default_factory=EconomicConfig)

    @classmethod
    def from_json(cls, filepath: str) -> "WindFarmConfig":
        """从JSON文件加载配置。"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        opt_config = OptimizationConfig(**data.get("optimization", {}))
        vis_config = VisualizationConfig(**data.get("visualization", {}))
        econ_config = EconomicConfig(**data.get("economic", {}))

        return cls(
            n_turbines=data.get("n_turbines", 15),
            turbine_model=data.get("turbine_model", "V126-3.45MW"),
            wake_model=data.get("wake_model", "jensen"),
            wake_decay=data.get("wake_decay", 0.07),
            superposition_method=data.get("superposition_method", "sum_of_squares"),
            boundary_type=data.get("boundary_type", "rectangular"),
            boundary_params=data.get("boundary_params", {}),
            exclusions=list(data.get("exclusions", [])),
            wind_resource_type=data.get("wind_resource_type", "default"),
            wind_resource_params=data.get("wind_resource_params", {}),
            optimization=opt_config,
            visualization=vis_config,
            economic=econ_config,
        )

    def to_json(self, filepath: str) -> None:
        """保存配置到JSON文件。"""
        data = {
            "n_turbines": self.n_turbines,
            "turbine_model": self.turbine_model,
            "wake_model": self.wake_model,
            "wake_decay": self.wake_decay,
            "superposition_method": self.superposition_method,
            "boundary_type": self.boundary_type,
            "boundary_params": self.boundary_params,
            "exclusions": self.exclusions,
            "wind_resource_type": self.wind_resource_type,
            "wind_resource_params": self.wind_resource_params,
            "optimization": self.optimization.__dict__,
            "visualization": self.visualization.__dict__,
            "economic": self.economic.__dict__,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def create_turbines(self) -> list[Turbine]:
        """根据配置创建风机列表。"""
        turbine = create_default_turbine(self.turbine_model)
        return [turbine for _ in range(self.n_turbines)]

    def create_wake_model(self) -> WakeModel:
        """根据配置创建尾流模型。"""
        if self.wake_model.lower() == "jensen":
            return JensenWake(wake_decay=self.wake_decay)
        elif self.wake_model.lower() == "gaussian":
            return GaussianWake(wake_decay=0.035)
        else:
            raise ValueError(f"未知的尾流模型: {self.wake_model}")

    def create_boundary(self) -> SiteBoundary:
        """根据配置创建场地边界。"""
        bp = self.boundary_params
        if self.boundary_type.lower() == "rectangular":
            return create_rectangular_boundary(
                width=bp.get("width", 4000),
                height=bp.get("height", 4000),
                center_x=bp.get("center_x", 0),
                center_y=bp.get("center_y", 0),
            )
        elif self.boundary_type.lower() == "hexagonal":
            return create_hexagonal_boundary(
                radius=bp.get("radius", 2500),
                center_x=bp.get("center_x", 0),
                center_y=bp.get("center_y", 0),
            )
        elif self.boundary_type.lower() == "irregular":
            return create_irregular_boundary()
        elif self.boundary_type.lower() == "custom":
            vertices = np.array(bp["vertices"], dtype=np.float64)
            return SiteBoundary(vertices)
        else:
            raise ValueError(f"未知的边界类型: {self.boundary_type}")

    def create_geofence(self, rotor_radius: Optional[float] = None) -> Geofence:
        """根据配置创建可行域（租赁边界 + 禁建区及其净距）。

        Parameters
        ----------
        rotor_radius : Optional[float]
            风轮半径 (m)。缺省时按配置机型取转子直径的一半，使禁建净距
            约束施加在风轮外缘而非塔架中心。
        """
        boundary = self.create_boundary()

        if rotor_radius is None:
            rotor_radius = float(create_default_turbine(self.turbine_model).rotor_diameter) / 2.0

        zones: list[ExclusionZone] = []
        used_names: set[str] = set()
        for k, raw in enumerate(self.exclusions):
            label = f"禁建区[{k}]"
            if not isinstance(raw, dict):
                raise ValueError(f"{label}: 配置项必须是对象，实际为 {type(raw).__name__}")

            vertices = raw.get("vertices")
            if vertices is None:
                raise ValueError(f"{label}: 缺少 'vertices' 顶点列表")
            try:
                verts = np.array(vertices, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label}: 'vertices' 无法解析为数值坐标数组: {exc}") from None

            setback = raw.get("setback", 0.0)
            try:
                setback = float(setback)
            except (TypeError, ValueError):
                raise ValueError(f"{label}: 'setback' 净距必须是数值 (m)") from None

            name = str(raw.get("name") or f"禁建区_{k + 1}")
            if name in used_names:
                raise ValueError(f"{label}: 禁建区名称 {name!r} 重复")
            used_names.add(name)

            # ExclusionZone 构造时会完成严格几何校验；Geofence 构造时
            # 还会校验禁建顶点均在租赁边界内。
            zones.append(
                ExclusionZone(
                    vertices=verts,
                    setback=setback,
                    name=name,
                    rotor_radius=rotor_radius,
                )
            )

        return Geofence(boundary=boundary, exclusions=zones)

    def create_wind_resource(self) -> WindResource:
        """根据配置创建风资源。"""
        wrp = self.wind_resource_params
        if self.wind_resource_type.lower() == "default":
            return create_default_wind_resource(
                num_sectors=wrp.get("num_sectors", 12),
                dominant_direction=wrp.get("dominant_direction", 270.0),
                mean_speed=wrp.get("mean_speed", 8.5),
            )
        elif self.wind_resource_type.lower() == "uniform":
            from .core.wind_resource import create_simple_wind_resource
            return create_simple_wind_resource(
                num_sectors=wrp.get("num_sectors", 12),
                uniform=True,
                mean_speed=wrp.get("mean_speed", 8.0),
            )
        else:
            raise ValueError(f"未知的风资源类型: {self.wind_resource_type}")


def create_sample_config() -> WindFarmConfig:
    """创建示例配置。

    示例场地为 3.5 km × 3.5 km 租赁区，内含三类最新勘测图新增的禁建区：
    航道（矩形）、海缆走廊（细长带状）与生态缓冲区（凹陷多边形），
    各自配置不同的风轮外缘安全净距。
    """
    return WindFarmConfig(
        n_turbines=12,
        turbine_model="V126-3.45MW",
        wake_model="jensen",
        wake_decay=0.07,
        boundary_type="rectangular",
        boundary_params={"width": 3500, "height": 3500, "center_x": 0, "center_y": 0},
        exclusions=[
            {
                "name": "航道",
                "setback": 200.0,
                "vertices": [
                    [-1750.0, -400.0],
                    [1750.0, -300.0],
                    [1750.0, 0.0],
                    [-1750.0, -100.0],
                ],
            },
            {
                "name": "海缆走廊",
                "setback": 100.0,
                "vertices": [
                    [600.0, -1750.0],
                    [750.0, -1750.0],
                    [750.0, 1750.0],
                    [600.0, 1750.0],
                ],
            },
            {
                # L 形凹陷多边形：生态缓冲区
                "name": "生态缓冲区",
                "setback": 150.0,
                "vertices": [
                    [-1600.0, 700.0],
                    [-900.0, 700.0],
                    [-900.0, 1100.0],
                    [-500.0, 1100.0],
                    [-500.0, 1600.0],
                    [-1600.0, 1600.0],
                ],
            },
        ],
        wind_resource_type="default",
        wind_resource_params={"num_sectors": 12, "dominant_direction": 270.0, "mean_speed": 8.5},
        optimization=OptimizationConfig(
            algorithm="ga",
            population_size=30,
            max_iterations=50,
            min_spacing_multiple=5.0,
            seed=42,
        ),
    )
