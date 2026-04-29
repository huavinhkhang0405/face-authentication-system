import time
import os
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
import joblib
from tqdm import tqdm
from src.model.coa import COA
from src.utils.config import DATA_DIR, MODEL_DIR, RESULT_DIR


# ======== Tải dữ liệu embedding ========
X = np.load(os.path.join(DATA_DIR, "embeddings.npy"))
y = np.load(os.path.join(DATA_DIR, "labels.npy"))

# Tự chia train / val / test nếu chưa có file riêng
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")

# Chuẩn hóa embedding để phù hợp với metric cosine
X_train = normalize(X_train)
X_val = normalize(X_val)
X_test = normalize(X_test)

# ======== Cấu hình tìm kiếm ========
max_k = max(3, min(10, len(X_train) - 2))  # Giới hạn k nhỏ hơn số mẫu
bounds = [
    (3, max_k),
    (0, 1),
    (0, 1)
]

var_types = ["int", "int"]
metric_list = ["euclidean", "cosine"]

print(f"COA dims = {len(bounds)} (k + weight + metric)")

# ======== Hàm fitness (độ chính xác nghịch đảo) ========
def fitness_wrapper(pop, cv_splits=3):
    fitness_vals = []
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    for sol in pop:
        k = max(3, int(round(sol[0])))
        weight = "uniform" if int(round(sol[1])) == 0 else "distance"
        metric = metric_list[int(round(sol[2])) % len(metric_list)]

        model = KNeighborsClassifier(
            n_neighbors=k,
            weights=weight,
            metric=metric,
            n_jobs=-1
        )

        try:
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
            mean_acc = scores.mean()
        except Exception as e:
            print("⚠️ Lỗi khi đánh giá:", e)
            mean_acc = 0.0

        fitness_vals.append(1 - mean_acc)
    return np.array(fitness_vals)

# ======== Khởi tạo và chạy COA ========
n_agents = 20
iterations = 20

coa = COA(
    func=fitness_wrapper,
    bounds=bounds,
    population_size=n_agents,
    iterations=iterations,
    var_types=var_types,
    minimize=True,
    early_stopping=True,
    patience=8,
    delta=1e-3
)

print("🚀 Running COA hyperparameter search ...")
start = time.time()
best_solution, best_fitness, curve = coa.run()
elapsed = time.time() - start
print(f"✅ COA done in {elapsed:.1f}s; best fitness (1-acc) = {best_fitness:.4f}")

# ======== Lọc các ứng viên tốt nhất ========
pop = coa.population
fitness_vals = fitness_wrapper(pop)
idx_sorted = np.argsort(fitness_vals)
top_n = 10
top_idx = idx_sorted[:top_n]
candidate_sols = [pop[i] for i in top_idx]
candidate_sols.insert(0, best_solution)

print("🔍 Refining top candidates on validation set...")
best_global_acc = -1
best_global_cfg = None
best_global_model = None
logs = []

for sol in tqdm(candidate_sols, desc="Refine", leave=True):
    k = max(3, int(round(sol[0])))
    weight = "uniform" if int(round(sol[1])) == 0 else "distance"
    metric = metric_list[int(round(sol[2])) % len(metric_list)]

    model = KNeighborsClassifier(n_neighbors=k, weights=weight, metric=metric, n_jobs=-1)
    model.fit(X_train, y_train)
    val_acc = model.score(X_val, y_val)

    cfg = {"k": k, "weight": weight, "metric": metric}
    logs.append((cfg, val_acc))
    print(f"Candidate: {cfg} --> Val acc = {val_acc:.4f}")

    if val_acc > best_global_acc:
        best_global_acc = val_acc
        best_global_cfg = cfg
        best_global_model = model

# ======== Huấn luyện lại trên toàn bộ train+val ========
X_full_train = np.vstack([X_train, X_val])
y_full_train = np.hstack([y_train, y_val])

final_model = KNeighborsClassifier(
    n_neighbors=best_global_cfg["k"],
    weights=best_global_cfg["weight"],
    metric=best_global_cfg["metric"],
    n_jobs=-1
)
final_model.fit(X_full_train, y_full_train)

test_acc = final_model.score(X_test, y_test)
print(f"🎯 Final model test accuracy = {test_acc:.4f}")

# ======== Lưu mô hình và kết quả ========
joblib.dump(final_model, os.path.join(MODEL_DIR, "best_knn_faceid.pkl"))
with open(os.path.join(RESULT_DIR, "best_params_faceid.txt"), "w") as f:
    f.write(f"{best_global_cfg},{best_global_acc:.4f}\n")
with open(os.path.join(RESULT_DIR, "top_candidates_faceid.txt"), "w") as f:
    for cfg, acc in logs:
        f.write(f"{cfg} --> {acc:.4f}\n")
np.save(os.path.join(RESULT_DIR, "coa_convergence_faceid.npy"), np.array(curve))

print("🏁 Best config:", best_global_cfg, "| Val Acc:", best_global_acc)
print("📈 Test Acc:", test_acc)
print("📦 Saved model and results.")
