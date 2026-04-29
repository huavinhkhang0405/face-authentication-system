import ast
import os
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from src.utils.config import MODEL_DIR, DATA_DIR, RESULT_DIR

# Đọc embedding
X = np.load(os.path.join(DATA_DIR, "embeddings.npy"))
y = np.load(os.path.join(DATA_DIR, "labels.npy"))

# Đọc tham số tối ưu
with open(os.path.join(RESULT_DIR,"best_params_faceid.txt"), "r") as f:
    line = f.readline().strip()
    params_str = line.split("},")[0] + "}"
    best_params = ast.literal_eval(params_str)

print("🔧 Best hyperparameters:", best_params)

# Huấn luyện mô hình với tham số tối ưu
knn = KNeighborsClassifier(
    n_neighbors=best_params["k"],
    weights=best_params["weight"],
    metric=best_params["metric"],
    n_jobs=-1
)

knn.fit(X, y)
joblib.dump(knn, os.path.join(MODEL_DIR, "best_knn_faceid.pkl"))
print("✅ Model trained and saved successfully!")
