import os
from src.config import DATA_DIR, RAW_DIR, ALIGNED_DIR

encrypted_raw = os.path.join(DATA_DIR, "encrypted_data", "raw")
encrypted_aligned = os.path.join(DATA_DIR, "encrypted_data", "faces_aligned")

print("=" * 60)
print("KIỂM TRA THỦ MỤC")
print("=" * 60)

print(f"\n📁 RAW_DIR: {RAW_DIR}")
print(f"   Exists: {os.path.exists(RAW_DIR)}")
if os.path.exists(RAW_DIR):
    content = os.listdir(RAW_DIR)
    print(f"   Content ({len(content)}): {content}")

print(f"\n📁 ALIGNED_DIR: {ALIGNED_DIR}")
print(f"   Exists: {os.path.exists(ALIGNED_DIR)}")
if os.path.exists(ALIGNED_DIR):
    content = os.listdir(ALIGNED_DIR)
    print(f"   Content ({len(content)}): {content}")

print(f"\n🔒 Encrypted Raw: {encrypted_raw}")
print(f"   Exists: {os.path.exists(encrypted_raw)}")
if os.path.exists(encrypted_raw):
    content = os.listdir(encrypted_raw)
    print(f"   Content ({len(content)}): {content}")

print(f"\n🔒 Encrypted Aligned: {encrypted_aligned}")
print(f"   Exists: {os.path.exists(encrypted_aligned)}")
if os.path.exists(encrypted_aligned):
    content = os.listdir(encrypted_aligned)
    print(f"   Content ({len(content)}): {content}")

print("\n" + "=" * 60)
