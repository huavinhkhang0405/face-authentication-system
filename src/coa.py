import numpy as np
import random

def _repair_int_bounds(pop, lb, ub, var_types):
    """Đảm bảo các biến trong giới hạn và làm tròn nếu cần."""
    pop = np.clip(pop, lb, ub)
    for j, vtype in enumerate(var_types):
        if vtype == "int":
            pop[:, j] = np.round(pop[:, j])
        elif vtype == "binary":
            pop[:, j] = np.round(pop[:, j]) % 2
    return pop


class COA:
    def __init__(self, func, bounds, population_size=20, iterations=50,
                 minimize=True, var_types=None,
                 early_stopping=False, patience=10, delta=1e-4):
        self.func = func
        self.lb = np.array([b[0] for b in bounds])
        self.ub = np.array([b[1] for b in bounds])
        self.pop_size = population_size
        self.iterations = iterations
        self.dimensions = len(bounds)
        self.minimize = minimize

        if var_types is None:
            self.var_types = ["int"] * self.dimensions
        else:
            self.var_types = var_types

        # Early stopping
        self.early_stopping = early_stopping
        self.patience = patience
        self.delta = delta

        # Khởi tạo quần thể ngẫu nhiên
        self.population = np.random.uniform(self.lb, self.ub,
                                            (self.pop_size, self.dimensions))
        self.population = _repair_int_bounds(self.population,
                                             self.lb, self.ub, self.var_types)

    def _is_better(self, f1, f2):
        """So sánh fitness theo hướng minimize/maximize."""
        return f1 < f2 if self.minimize else f1 > f2

    def update_positions(self, best_solution):
        """Cập nhật vị trí cá thể trong quần thể (coyote update)."""
        r = random.random()
        exploration_factor = random.randint(1, 2)
        new_pop = self.population.astype(float).copy()

        # Giữ cá thể tốt nhất (elitism)
        new_pop[0] = best_solution.copy()

        for i in range(1, len(new_pop)):
            candidate = new_pop[i]

            if i >= len(new_pop) // 2:
                # 🔹 Exploration phase (thăm dò ngẫu nhiên)
                random_vec = np.random.uniform(self.lb, self.ub, self.dimensions)
                candidate = candidate + r * (random_vec - exploration_factor * candidate)
            else:
                # 🔹 Exploitation phase (khai thác quanh best)
                candidate = candidate + r * (best_solution - exploration_factor * candidate)

            # Thêm nhiễu Gaussian nhỏ để tránh hội tụ sớm
            candidate += np.random.normal(0, 0.05, self.dimensions)

            new_pop[i] = candidate

        # Sửa lại để trong giới hạn và làm tròn nếu cần
        self.population = _repair_int_bounds(new_pop, self.lb, self.ub, self.var_types)

    def run(self):
        """Chạy quá trình tối ưu hóa chính."""
        best_solution = None
        best_fitness = float("inf") if self.minimize else -float("inf")
        curve = []
        no_improve_count = 0

        for iter_idx in range(self.iterations):
            print(f"COA iteration {iter_idx+1}/{self.iterations} ...")

            # Tính fitness toàn quần thể 1 lần duy nhất
            fitness = self.func(self.population)
            if fitness.ndim > 1:
                fitness = fitness.flatten()

            # Tìm cá thể tốt nhất hiện tại
            best_idx = np.argmin(fitness) if self.minimize else np.argmax(fitness)
            current_best = fitness[best_idx]

            # Cập nhật nếu tốt hơn trước
            if (self.minimize and current_best < best_fitness - self.delta) or \
               (not self.minimize and current_best > best_fitness + self.delta):
                best_fitness = current_best
                best_solution = self.population[best_idx].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            curve.append(best_fitness)
            print(f"  ↳ Best fitness: {best_fitness:.5f} | No improve: {no_improve_count}")

            # Cập nhật vị trí quần thể
            if best_solution is not None:
                self.update_positions(best_solution)

            # 🔹 Kiểm tra dừng sớm
            if self.early_stopping and no_improve_count >= self.patience:
                print(f"Early stopping at iteration {iter_idx+1}")
                break

        return best_solution, best_fitness, curve
