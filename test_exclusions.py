"""禁建区/可行域端到端验证脚本（不纳入正式测试套件）。"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from wind_farm_opt.constraints import (
    SiteBoundary,
    Geofence,
    ExclusionZone,
    InfeasibleLayoutError,
    validate_polygon_geometry,
)
from wind_farm_opt.constraints.spacing import (
    enforce_min_spacing,
    compute_min_spacing_from_diameters,
    check_min_spacing,
)
from wind_farm_opt.optimization.baseline import generate_grid_layout
from wind_farm_opt.optimization.ga import GeneticAlgorithm, GAConfig
from wind_farm_opt.optimization.pso import ParticleSwarmOptimizer, PSOConfig
from wind_farm_opt.config import create_sample_config


def header(t):
    print(f"\n=== {t} ===")


failures = []


def check(name, cond):
    print(("  ✓ " if cond else "  ✗ ") + name)
    if not cond:
        failures.append(name)


# 1. 几何严格校验
header("1. 几何数据严格校验")

# 合法凹多边形（L 形）
concave = np.array([
    [0.0, 0.0], [1000.0, 0.0], [1000.0, 400.0],
    [600.0, 400.0], [600.0, 1000.0], [0.0, 1000.0],
])
validate_polygon_geometry(concave, "L形")
check("合法凹多边形通过", True)

bad_cases = [
    ("顶点数不足", np.array([[0.0, 0], [1, 1]])),
    ("NaN", np.array([[0.0, 0], [1, np.nan], [2, 0]])),
    ("重复点", np.array([[0.0, 0], [1, 0], [1, 0], [0, 1]])),
    ("零面积", np.array([[0.0, 0], [1, 1], [2, 2]])),
    ("自相交(蝴蝶结)", np.array([[0.0, 0], [1000, 1000], [1000, 0], [0, 1000]])),
]
for label, verts in bad_cases:
    try:
        validate_polygon_geometry(verts, label)
        check(f"拒绝{label}", False)
    except ValueError:
        check(f"拒绝{label}", True)

try:
    ExclusionZone(vertices=concave, setback=-10, name="坏净距")
    check("拒绝负净距", False)
except ValueError:
    check("拒绝负净距", True)

# 2. 凹多边形包含判定
header("2. 凹多边形点包含")
b = SiteBoundary(concave, "L形场地")
check("凹口内部点在域内", b.contains_point(np.array([200.0, 600.0])))
check("凹口挖空区域点在域外", not b.contains_point(np.array([800.0, 700.0])))
check("域外点判定正确", not b.contains_point(np.array([-10.0, 500.0])))

# 3. 禁建净距 + 可行域采样
header("3. 禁建区净距与可行域采样")
lease = SiteBoundary(np.array([
    [0.0, 0], [4000, 0], [4000, 4000], [0, 4000]
]), "租赁区")
zone = ExclusionZone(
    vertices=np.array([[1800.0, 1800], [2200, 1800], [2200, 2200], [1800, 2200]]),
    setback=200.0, name="航道", rotor_radius=63.0,
)
site = Geofence(boundary=lease, exclusions=[zone])

rng = np.random.default_rng(0)
pts = site.sample_feasible_points(300, rng, max_attempts=30000)
dists = np.array([zone.point_distance(p) for p in pts])
check("采样点全部在租赁边界内", lease.contains_all(pts).all())
check(f"采样点全部满足中心净距 ≥ {zone.center_setback:.0f} m",
      bool((dists + 1e-7 >= zone.center_setback).all()))

# 点在禁建区内 -> 违反深度 = center_setback
depth_inside = zone.violation_depth(np.array([2000.0, 2000]))
check("区内点违反深度=中心净距", abs(depth_inside - zone.center_setback) < 1e-9)
depth_ok = zone.violation_depth(np.array([2000.0, 3000]))
check("远离点违反深度为0", depth_ok == 0.0)

# 4. 投影回可行域
header("4. 违规点投影回可行域")
p_outside = np.array([4500.0, 2000.0])
p1 = site.project_into_feasible_region(p_outside, rng)
check("界外点投影后可行", site.is_feasible_point(p1))
p_inside_zone = np.array([2000.0, 2000.0])
p2 = site.project_into_feasible_region(p_inside_zone, rng)
check("禁建区内点投影后可行", site.is_feasible_point(p2))

# 5. 基线 + 间距修复
header("5. 规则基线（含禁建区）")
diam = np.full(12, 126.0)
min_spacing = compute_min_spacing_from_diameters(diam, 5.0)
layout = generate_grid_layout(site, 12, diam, min_multiple=5.0, rng=rng)
check("基线机位全部可行", bool(site.feasible_mask(layout).all()))
valid, _ = check_min_spacing(layout, min_spacing)
check("基线满足最小间距", valid)

# 6. GA / PSO 惩罚与修复
header("6. GA / PSO 优化结果可行")
from wind_farm_opt.core.turbine import create_default_turbine
from wind_farm_opt.core.wind_resource import create_default_wind_resource
from wind_farm_opt.core.wake import JensenWake
from wind_farm_opt.farm.aep import AEPCalculator

turbines = [create_default_turbine("V126-3.45MW") for _ in range(12)]
wr = create_default_wind_resource()
aep = AEPCalculator(turbines, wr, JensenWake(0.07), "sum_of_squares", speed_step=2.0)

t0 = time.time()
ga = GeneticAlgorithm(12, diam, site, aep.evaluate_layout,
                      GAConfig(population_size=8, max_generations=4, seed=1))
res = ga.optimize(verbose=False)
check("GA 返回可行解", bool(site.feasible_mask(res.best_positions).all())
      and check_min_spacing(res.best_positions, min_spacing)[0])
print(f"    GA 耗时 {time.time()-t0:.1f}s, best={res.best_fitness:.0f} MWh")

t0 = time.time()
pso = ParticleSwarmOptimizer(12, diam, site, aep.evaluate_layout,
                             PSOConfig(swarm_size=8, max_iterations=4, seed=1))
res2 = pso.optimize(verbose=False)
check("PSO 返回可行解", bool(site.feasible_mask(res2.best_positions).all())
      and check_min_spacing(res2.best_positions, min_spacing)[0])
print(f"    PSO 耗时 {time.time()-t0:.1f}s, best={res2.best_fitness:.0f} MWh")

# 7. 密集 / 根本不可行场景：有限时间失败 + 具体原因
header("7. 不可行场景快速失败并给出约束原因")

# 7a. 禁建区覆盖绝大部分场地
big_zone = ExclusionZone(
    vertices=np.array([[100.0, 100], [3900, 100], [3900, 3900], [100, 3900]]),
    setback=300.0, name="巨型禁建区", rotor_radius=63.0,
)
tiny_site = Geofence(boundary=lease, exclusions=[big_zone])
t0 = time.time()
try:
    tiny_site.sample_feasible_points(5, np.random.default_rng(2), max_attempts=3000)
    check("不可行采样抛错", False)
except InfeasibleLayoutError as e:
    check("不可行采样抛错", True)
    check("失败信息含命中率", "命中率" in str(e))
print(f"    采样失败耗时 {time.time()-t0:.2f}s")

# 7b. 机位过密：基线必须带原因失败
t0 = time.time()
try:
    generate_grid_layout(site, 200, diam, min_multiple=5.0,
                         rng=np.random.default_rng(3))
    check("过密基线抛错", False)
except InfeasibleLayoutError as e:
    check("过密基线抛错", True)
    print(f"    原因: {e}")
print(f"    过密判定耗时 {time.time()-t0:.1f}s")

# 7c. 优化器在不可行场地初始化时失败
try:
    bad_ga = GeneticAlgorithm(12, diam, tiny_site, aep.evaluate_layout,
                              GAConfig(population_size=4, max_generations=2, seed=4))
    bad_ga.optimize(verbose=False)
    check("不可行 GA 初始化失败", False)
except InfeasibleLayoutError as e:
    check("不可行 GA 初始化失败", True)
    print(f"    原因: {e}")

# 7d. 诊断报告
bad_layout = np.array([[2000.0, 2000.0], [2001.0, 2001.0], [4500.0, 0.0]])
viol = site.diagnose(bad_layout, min_spacing)
check("诊断覆盖禁建区/间距/出界三类",
      {v.constraint for v in viol} >= {"航道", "min_spacing", "lease_boundary"})
report = InfeasibleLayoutError("演示", violations=viol).violation_report()
check("失败报告含具体风机编号", "#0" in report and "#2" in report)
print("\n" + report)

# 8. 绘图区分三类要素
header("8. 可视化")
from wind_farm_opt.visualization.plotting import plot_farm_layout, plot_wake_heatmap

os.makedirs("test_output", exist_ok=True)
cfg = create_sample_config()
sample_site = cfg.create_geofence()
sample_layout = generate_grid_layout(sample_site, 12, diam, min_multiple=5.0,
                                     rng=np.random.default_rng(5))
check("示例场地基线可行", bool(sample_site.feasible_mask(sample_layout).all()))
losses = np.zeros(len(sample_layout))
plot_farm_layout(
    sample_layout, sample_site,
    diam, turbine_losses=losses,
    turbine_names=[f"#{i}" for i in range(len(sample_layout))],
    save_path="test_output/exclusion_layout.png",
)
check("布局图已生成", os.path.exists("test_output/exclusion_layout.png"))

plot_wake_heatmap(
    layout, site, JensenWake(0.07), 270.0, diam,
    np.full(12, 0.8), grid_resolution=60,
    save_path="test_output/exclusion_heatmap.png",
)
check("热力图已生成", os.path.exists("test_output/exclusion_heatmap.png"))

print("\n" + "=" * 60)
if failures:
    print(f"失败 {len(failures)} 项: {failures}")
    sys.exit(1)
print("全部禁建区/可行域验证通过 ✓")
print("=" * 60)
