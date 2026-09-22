"""粒子群优化器。"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..constraints.geofence import Geofence, InfeasibleLayoutError
from ..constraints.spacing import (
    check_min_spacing,
    compute_min_spacing_from_diameters,
    enforce_min_spacing,
)


@dataclass
class PSOConfig:
    """粒子群算法配置参数。

    Parameters
    ----------
    swarm_size : int
        粒子群大小
    max_iterations : int
        最大迭代次数
    inertia_weight : float
        惯性权重 w
    cognitive_coeff : float
        认知系数 c1
    social_coeff : float
        社会系数 c2
    max_velocity : float
        最大速度（占场地范围的比例）
    min_spacing_multiple : float
        最小间距倍数（相对于转子直径）
    penalty_factor : float
        约束违反惩罚因子
    seed : Optional[int]
        随机种子
    """

    swarm_size: int = 40
    max_iterations: int = 150
    inertia_weight: float = 0.7
    cognitive_coeff: float = 1.49
    social_coeff: float = 1.49
    max_velocity: float = 0.2
    min_spacing_multiple: float = 5.0
    penalty_factor: float = 1e6
    seed: Optional[int] = None


class ParticleSwarmOptimizer:
    """粒子群算法机位优化器。

    约束：最小间距、租赁边界内、与各禁建区保持各自的安全净距。
    """

    def __init__(
        self,
        n_turbines: int,
        rotor_diameters: np.ndarray,
        boundary,
        fitness_fn: Callable[[np.ndarray], float],
        config: Optional[PSOConfig] = None,
        site: Optional[Geofence] = None,
    ) -> None:
        self.n_turbines = n_turbines
        self.rotor_diameters = np.asarray(rotor_diameters, dtype=np.float64)

        if site is not None:
            self.site = site
        elif isinstance(boundary, Geofence):
            self.site = boundary
        else:
            self.site = Geofence(boundary=boundary)
        self.boundary = self.site.boundary
        self.fitness_fn = fitness_fn
        self.config = config if config is not None else PSOConfig()

        self.rng = np.random.default_rng(self.config.seed)

        self.min_spacing = compute_min_spacing_from_diameters(
            self.rotor_diameters,
            self.config.min_spacing_multiple,
        )

        self.n_dim = n_turbines * 2
        self.x_range = self.site.x_max - self.site.x_min
        self.y_range = self.site.y_max - self.site.y_min

        self.vel_range = np.zeros(self.n_dim, dtype=np.float64)
        for i in range(self.n_dim):
            self.vel_range[i] = (
                self.x_range if i % 2 == 0 else self.y_range
            ) * self.config.max_velocity

        self.pos_bounds = np.zeros((self.n_dim, 2), dtype=np.float64)
        for i in range(self.n_dim):
            if i % 2 == 0:
                self.pos_bounds[i] = [self.site.x_min, self.site.x_max]
            else:
                self.pos_bounds[i] = [self.site.y_min, self.site.y_max]

        self._best_global_pos = None
        self._best_global_fitness = -np.inf
        self._best_iteration = 0

        self.convergence_history: list[float] = []
        self.mean_history: list[float] = []

    def _initialize_swarm(self, swarm_size: int) -> tuple[np.ndarray, np.ndarray]:
        """初始化粒子群，初始粒子全部位于可行域内。"""
        positions = np.zeros((swarm_size, self.n_dim), dtype=np.float64)
        velocities = np.zeros((swarm_size, self.n_dim), dtype=np.float64)

        for i in range(swarm_size):
            positions[i] = self._generate_valid_layout().flatten()
            velocities[i] = self.rng.uniform(
                -self.vel_range, self.vel_range, self.n_dim
            )

        return positions, velocities

    def _generate_valid_layout(self) -> np.ndarray:
        """生成一个满足全部约束（含禁建净距）的初始布局。"""
        max_rounds = 60

        for _ in range(max_rounds):
            try:
                positions = self.site.sample_feasible_points(
                    self.n_turbines, self.rng, max_attempts=20000
                )
            except InfeasibleLayoutError:
                raise
            valid, _ = check_min_spacing(positions, self.min_spacing)
            if valid:
                return positions
            try:
                return enforce_min_spacing(
                    positions, self.min_spacing, self.site, self.rng
                )
            except InfeasibleLayoutError:
                continue

        raise InfeasibleLayoutError(
            f"无法在可行域内为 {self.n_turbines} 台风机生成满足 "
            f"{self.min_spacing:.0f} m 最小间距的初始布局",
            violations=[],
        )

    def _compute_penalty(self, positions_flat: np.ndarray) -> float:
        """计算约束违反惩罚（租赁边界 + 禁建净距 + 最小间距）。"""
        positions = positions_flat.reshape(self.n_turbines, 2)
        depth = self.site.total_violation_depth(positions, self.min_spacing)
        return depth * self.config.penalty_factor

    def _evaluate_particles(self, positions: np.ndarray) -> np.ndarray:
        """评估所有粒子的适应度。"""
        swarm_size = positions.shape[0]
        fitness = np.zeros(swarm_size, dtype=np.float64)

        for i in range(swarm_size):
            penalty = self._compute_penalty(positions[i])

            if penalty > 0:
                fitness[i] = -penalty
            else:
                pos_reshaped = positions[i].reshape(self.n_turbines, 2)
                try:
                    fitness[i] = self.fitness_fn(pos_reshaped)
                except Exception:
                    fitness[i] = -self.config.penalty_factor

        return fitness

    def _repair(self, positions_flat: np.ndarray) -> np.ndarray:
        """修复违反约束的粒子（投影回可行域 + 间距推开/重定位）。

        修复失败时保留原粒子，由结束前的最终可行性校验统一拒绝。
        """
        positions = positions_flat.reshape(self.n_turbines, 2)

        for i in range(self.n_turbines):
            if not self.site.is_feasible_point(positions[i]):
                try:
                    positions[i] = self.site.project_into_feasible_region(
                        positions[i], self.rng
                    )
                except InfeasibleLayoutError:
                    pass

        valid, _ = check_min_spacing(positions, self.min_spacing)
        feasible = self.site.feasible_mask(positions).all()

        if not (valid and feasible):
            try:
                positions = enforce_min_spacing(
                    positions, self.min_spacing, self.site, self.rng
                )
            except InfeasibleLayoutError:
                return positions_flat

        return positions.flatten()

    def optimize(self, verbose: bool = True) -> "OptimizeResult":
        """执行优化。

        Returns
        -------
        OptimizeResult
            优化结果
        """
        from .ga import OptimizeResult

        swarm_size = self.config.swarm_size
        max_iter = self.config.max_iterations

        w = self.config.inertia_weight
        c1 = self.config.cognitive_coeff
        c2 = self.config.social_coeff

        if verbose:
            print(f"\n=== 粒子群优化开始 ===")
            print(f"风机台数: {self.n_turbines}")
            print(f"粒子群大小: {swarm_size}")
            print(f"最大迭代: {max_iter}")
            print(f"最小间距: {self.min_spacing:.1f} m "
                  f"({self.config.min_spacing_multiple:.1f}倍转子直径)")
            print(f"w={w}, c1={c1}, c2={c2}")
            if self.site.has_exclusions:
                print(f"禁建区: {len(self.site.exclusions)} 个")
                for z in self.site.exclusions:
                    print(f"  - {z.name}: 外缘净距 {z.setback:.0f} m "
                          f"(中心限制 {z.center_setback:.0f} m)")
            print("=" * 35)

        positions, velocities = self._initialize_swarm(swarm_size)
        fitness = self._evaluate_particles(positions)

        best_personal_pos = positions.copy()
        best_personal_fitness = fitness.copy()

        best_global_idx = np.argmax(fitness)
        self._best_global_pos = positions[best_global_idx].reshape(self.n_turbines, 2).copy()
        self._best_global_fitness = float(fitness[best_global_idx])
        self._best_iteration = 0

        for iteration in range(max_iter):
            self.convergence_history.append(float(self._best_global_fitness))
            self.mean_history.append(float(np.mean(fitness)))

            r1 = self.rng.random((swarm_size, self.n_dim))
            r2 = self.rng.random((swarm_size, self.n_dim))

            best_global_flat = self._best_global_pos.flatten()

            velocities = (
                w * velocities
                + c1 * r1 * (best_personal_pos - positions)
                + c2 * r2 * (best_global_flat - positions)
            )

            velocities = np.clip(velocities, -self.vel_range, self.vel_range)

            positions = positions + velocities

            positions = np.clip(
                positions,
                self.pos_bounds[:, 0],
                self.pos_bounds[:, 1],
            )

            for i in range(swarm_size):
                positions[i] = self._repair(positions[i])

            fitness = self._evaluate_particles(positions)

            improved_mask = fitness > best_personal_fitness
            best_personal_pos[improved_mask] = positions[improved_mask].copy()
            best_personal_fitness[improved_mask] = fitness[improved_mask].copy()

            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > self._best_global_fitness:
                self._best_global_fitness = float(fitness[current_best_idx])
                self._best_global_pos = positions[current_best_idx].reshape(
                    self.n_turbines, 2
                ).copy()
                self._best_iteration = iteration + 1

            if verbose and (iteration % 5 == 0 or iteration == max_iter - 1):
                print(
                    f"Iter {iteration+1:3d} | "
                    f"Best: {self._best_global_fitness/1e3:8.2f} GWh | "
                    f"Mean: {np.mean(fitness)/1e3:8.2f} GWh | "
                    f"Found@Iter {self._best_iteration}"
                )

        if verbose:
            print("=" * 35)
            print(f"优化完成!")
            print(f"最优净AEP: {self._best_global_fitness/1e3:.2f} GWh")
            print(f"找到最优解的迭代: {self._best_iteration}")

        best_positions = self._best_global_pos.copy()
        violations = self.site.diagnose(best_positions, self.min_spacing)
        feasible = len(violations) == 0

        if not feasible:
            raise InfeasibleLayoutError(
                "粒子群算法未能在有限迭代内找到满足全部约束的布局"
                "（租赁边界、禁建净距、最小间距）",
                violations=violations,
            )

        return OptimizeResult(
            best_positions=best_positions,
            best_fitness=float(self._best_global_fitness),
            best_generation=self._best_iteration,
            convergence_history=self.convergence_history.copy(),
            mean_history=self.mean_history.copy(),
            final_population=positions.copy(),
            final_fitness=fitness.copy(),
            feasible=feasible,
        )
