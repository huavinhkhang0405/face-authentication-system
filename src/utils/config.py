import os
import torch
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
ALIGNED_DIR = os.path.join(DATA_DIR, "faces_aligned")

EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")
LABELS_FILE = os.path.join(DATA_DIR, "labels.npy")

MODEL_DIR = os.path.join(BASE_DIR, "models")
KNN_FILE = os.path.join(MODEL_DIR, "knn.pkl")
LABEL_ENCODER_FILE = os.path.join(MODEL_DIR, "label_encoder.pkl")
META_FILE = os.path.join(MODEL_DIR, "meta.json")

RESULT_DIR = os.path.join(BASE_DIR, "results")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 160

# File lưu khóa mã hóa (QUAN TRỌNG: Lưu trữ an toàn!)
KEY_FILE = os.path.join(BASE_DIR, "encryption.key")
