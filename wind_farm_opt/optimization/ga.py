"""遗传算法优化器。"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..constraints.exclusion import (
    FeasibleDomain,
    InfeasibleLayoutError,
)
from ..constraints.spacing import (
    compute_min_spacing_from_diameters,
)


@dataclass
class GAConfig:
    """遗传算法配置参数。

    Parameters
    ----------
    population_size : int
        种群大小
    max_generations : int
        最大迭代代数
    crossover_rate : float
        交叉概率
    mutation_rate : float
        变异概率
    mutation_strength : float
        变异强度（坐标标准差占场地范围的比例）
    elite_ratio : float
        精英保留比例
    tournament_size : int
        锦标赛选择的规模
    min_spacing_multiple : float
        最小间距倍数（相对于转子直径）
    penalty_factor : float
        约束违反惩罚因子
    seed : Optional[int]
        随机种子
    """

    population_size: int = 50
    max_generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.15
    mutation_strength: float = 0.1
    elite_ratio: float = 0.1
    tournament_size: int = 3
    min_spacing_multiple: float = 5.0
    penalty_factor: float = 1e6
    seed: Optional[int] = None
    init_time_budget_s: float = 30.0


@dataclass
class OptimizeResult:
    """优化结果。

    Parameters
    ----------
    best_positions : np.ndarray
        最优风机位置 (N_turb, 2)
    best_fitness : float
        最优适应度（净AEP，MWh/year）
    best_generation : int
        找到最优解的代数
    convergence_history : list[float]
        每代最优适应度历史
    mean_history : list[float]
        每代平均适应度历史
    final_population : np.ndarray
        最终种群 (pop_size, N_turb*2)
    final_fitness : np.ndarray
        最终种群适应度 (pop_size,)
    """

    best_positions: np.ndarray
    best_fitness: float
    best_generation: int
    convergence_history: list[float]
    mean_history: list[float]
    final_population: np.ndarray
    final_fitness: np.ndarray
    feasible: bool = True
    violation_report: Optional[dict] = None
    failure_reasons: Optional[list[str]] = None


class GeneticAlgorithm:
    """遗传算法机位优化器。

    优化目标：最大化年净发电量（等价于最小化尾流损失）。
    约束：最小间距、租赁边界内、远离所有禁建区及其安全净距。
    """

    def __init__(
        self,
        n_turbines: int,
        rotor_diameters: np.ndarray,
        domain: FeasibleDomain,
        fitness_fn: Callable[[np.ndarray], float],
        config: Optional[GAConfig] = None,
    ) -> None:
        """
        Parameters
        ----------
        n_turbines : int
            风机台数
        rotor_diameters : np.ndarray
            每台风机的转子直径
        domain : FeasibleDomain
            可行域（租赁边界 − 禁建区及各自净距）
        fitness_fn : Callable[[np.ndarray], float]
            适应度函数，输入位置数组 (N_turb, 2)，返回净AEP
        config : Optional[GAConfig]
            算法配置参数
        """
        self.n_turbines = n_turbines
        self.rotor_diameters = np.asarray(rotor_diameters, dtype=np.float64)
        self.domain = domain
        self.boundary = domain.lease
        self.fitness_fn = fitness_fn
        self.config = config if config is not None else GAConfig()

        self.rng = np.random.default_rng(self.config.seed)

        self.min_spacing = compute_min_spacing_from_diameters(
            self.rotor_diameters,
            self.config.min_spacing_multiple,
        )

        self.n_dim = n_turbines * 2
        self.x_range = self.boundary.x_max - self.boundary.x_min
        self.y_range = self.boundary.y_max - self.boundary.y_min

        self._best_positions = None
        self._best_fitness = -np.inf
        self._best_generation = 0

        self.convergence_history: list[float] = []
        self.mean_history: list[float] = []

    def _initialize_population(self, pop_size: int) -> np.ndarray:
        """初始化种群，每个个体都是满足全部约束的可行布局。"""
        population = np.zeros((pop_size, self.n_dim), dtype=np.float64)

        for i in range(pop_size):
            positions = self._generate_valid_layout()
            population[i] = positions.flatten()

        return population

    def _generate_valid_layout(self) -> np.ndarray:
        """生成一个满足租赁边界、禁建净距与间距约束的初始布局。"""
        try:
            return self.domain.sample_separated_points(
                self.n_turbines,
                self.min_spacing,
                self.rng,
                time_budget_s=self.config.init_time_budget_s,
            )
        except InfeasibleLayoutError:
            raise
        except Exception as exc:  # 采样器之外的意外错误统一转换
            raise InfeasibleLayoutError(
                "无法生成满足约束的初始布局",
                reasons=[str(exc)],
            ) from exc

    def _compute_penalty(self, positions_flat: np.ndarray) -> tuple[float, dict]:
        """计算租赁边界/禁建净距/间距的连续距离惩罚。"""
        positions = positions_flat.reshape(self.n_turbines, 2)
        return self.domain.penalty(
            positions,
            min_spacing=self.min_spacing,
            factor=self.config.penalty_factor,
        )

    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """评估整个种群的适应度（带约束惩罚）。"""
        pop_size = population.shape[0]
        fitness = np.zeros(pop_size, dtype=np.float64)

        for i in range(pop_size):
            positions = population[i].reshape(self.n_turbines, 2)

            penalty, _metrics = self._compute_penalty(population[i])

            if penalty > 0:
                fitness[i] = -penalty
            else:
                try:
                    fitness[i] = self.fitness_fn(positions)
                except Exception:
                    fitness[i] = -self.config.penalty_factor

        return fitness

    def _tournament_selection(
        self, population: np.ndarray, fitness: np.ndarray, n_select: int
    ) -> np.ndarray:
        """锦标赛选择。"""
        pop_size = population.shape[0]
        selected = np.zeros((n_select, self.n_dim), dtype=np.float64)

        for i in range(n_select):
            candidates = self.rng.integers(0, pop_size, size=self.config.tournament_size)
            best_idx = candidates[np.argmax(fitness[candidates])]
            selected[i] = population[best_idx]

        return selected

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """均匀交叉。"""
        if self.rng.random() > self.config.crossover_rate:
            return parent1.copy()

        mask = self.rng.integers(0, 2, size=self.n_dim, dtype=bool)
        child = np.where(mask, parent1, parent2)

        return child

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """高斯变异。"""
        mutated = individual.copy()

        for i in range(self.n_dim):
            if self.rng.random() < self.config.mutation_rate:
                range_sigma = (
                    self.x_range if i % 2 == 0 else self.y_range
                ) * self.config.mutation_strength
                mutated[i] += self.rng.normal(0.0, range_sigma)

        return mutated

    def _repair(self, individual: np.ndarray) -> np.ndarray:
        """把交叉/变异后的个体修复回可行域。

        先逐点投影回可行域并推开间距不足的风机对；若有限迭代内无法
        修复，则用一个全新的可行随机布局替换该个体，保证种群中不存
        在违规布局。
        """
        positions = individual.reshape(self.n_turbines, 2)
        try:
            repaired = self.domain.repair_layout(
                positions, min_spacing=self.min_spacing, rng=self.rng
            )
            return repaired.flatten()
        except InfeasibleLayoutError:
            return self._generate_valid_layout().flatten()

    def optimize(self, verbose: bool = True) -> OptimizeResult:
        """执行优化。

        Parameters
        ----------
        verbose : bool
            是否打印进度信息

        Returns
        -------
        OptimizeResult
            优化结果
        """
        pop_size = self.config.population_size
        max_gen = self.config.max_generations

        n_elite = max(1, int(pop_size * self.config.elite_ratio))

        if verbose:
            print(f"\n=== 遗传算法优化开始 ===")
            print(f"风机台数: {self.n_turbines}")
            print(f"种群大小: {pop_size}")
            print(f"最大代数: {max_gen}")
            print(f"最小间距: {self.min_spacing:.1f} m "
                  f"({self.config.min_spacing_multiple:.1f}倍转子直径)")
            print(f"租赁面积: {self.boundary.area / 1e6:.2f} km²")
            print(f"可行域面积: {self.domain.area / 1e6:.2f} km² "
                  f"（含 {len(self.domain.zones)} 个禁建区）")
            print("=" * 35)

        try:
            population = self._initialize_population(pop_size)
        except InfeasibleLayoutError as exc:
            if verbose:
                print("初始种群生成失败：布局问题不可行")
                for reason in exc.reasons:
                    print(f"  - {reason}")
            raise
        fitness = self._evaluate_population(population)

        best_idx = np.argmax(fitness)
        self._best_fitness = float(fitness[best_idx])
        self._best_positions = population[best_idx].reshape(self.n_turbines, 2)
        self._best_generation = 0

        for gen in range(max_gen):
            self.convergence_history.append(float(self._best_fitness))
            self.mean_history.append(float(np.mean(fitness)))

            elite_idx = np.argsort(fitness)[-n_elite:]
            elites = population[elite_idx].copy()

            parents = self._tournament_selection(population, fitness, pop_size - n_elite)

            offspring = np.zeros((pop_size - n_elite, self.n_dim), dtype=np.float64)
            for i in range(0, pop_size - n_elite, 2):
                p1 = parents[i]
                p2 = parents[(i + 1) % (pop_size - n_elite)]
                c1 = self._crossover(p1, p2)
                c2 = self._crossover(p2, p1)
                offspring[i] = self._mutate(c1)
                if i + 1 < pop_size - n_elite:
                    offspring[i + 1] = self._mutate(c2)

            for i in range(len(offspring)):
                offspring[i] = self._repair(offspring[i])

            population[:n_elite] = elites
            population[n_elite:] = offspring

            fitness = self._evaluate_population(population)

            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > self._best_fitness:
                self._best_fitness = float(fitness[current_best_idx])
                self._best_positions = population[current_best_idx].reshape(
                    self.n_turbines, 2
                ).copy()
                self._best_generation = gen + 1

            if verbose and (gen % 5 == 0 or gen == max_gen - 1):
                print(
                    f"Gen {gen+1:3d} | "
                    f"Best: {self._best_fitness/1e3:8.2f} GWh | "
                    f"Mean: {np.mean(fitness)/1e3:8.2f} GWh | "
                    f"Found@Gen {self._best_generation}"
                )

        if verbose:
            print("=" * 35)
            print(f"优化完成!")
            print(f"最优净AEP: {self._best_fitness/1e3:.2f} GWh")
            print(f"找到最优解的代数: {self._best_generation}")

        # 最终把关：只返回真正可行的布局，任何违规都在此显式失败。
        final_report = self.domain.validate_layout(
            self._best_positions, self.min_spacing
        )
        if not final_report.feasible:
            raise InfeasibleLayoutError(
                "GA 最终最优解仍违反约束",
                reasons=final_report.reasons,
                details={"report": final_report.to_dict()},
            )

        return OptimizeResult(
            best_positions=self._best_positions.copy(),
            best_fitness=float(self._best_fitness),
            best_generation=self._best_generation,
            convergence_history=self.convergence_history.copy(),
            mean_history=self.mean_history.copy(),
            final_population=population.copy(),
            final_fitness=fitness.copy(),
            feasible=True,
        )
