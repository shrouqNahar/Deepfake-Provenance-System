# utils.py
import os
import json
import hashlib
import random
import shutil
from datetime import datetime
from PIL import Image
import imagehash
from sklearn.model_selection import train_test_split
from config import DATASET_PATH

# ---------- Basic hashing ----------
def compute_sha256(file_path):
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def compute_phash(image_path):
    return str(imagehash.phash(Image.open(image_path)))

def hamming_distance(h1, h2):
    return imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(h2)

# ---------- Metadata generation ----------
def generate_metadata(image_path, use_phash=True):
    metadata = {
        "hash": compute_sha256(image_path),
        "timestamp": datetime.now().isoformat(),
        "device_id": hashlib.sha256(b"device_01").hexdigest()[:16]
    }
    if use_phash:
        metadata["phash"] = compute_phash(image_path)
    with open(image_path + ".json", "w") as f:
        json.dump(metadata, f)

def generate_metadata_for_folder(folder):
    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            generate_metadata(os.path.join(folder, file))

# ---------- Provenance manipulation ----------
def remove_random_metadata(folder, percentage=0.2):
    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    remove_count = int(len(files) * percentage)
    for f in random.sample(files, remove_count):
        os.remove(os.path.join(folder, f))

def forge_random_metadata(folder, percentage=0.1):
    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    forge_count = int(len(files) * percentage)
    for f in random.sample(files, forge_count):
        path = os.path.join(folder, f)
        with open(path, "r") as file:
            metadata = json.load(file)
        metadata["hash"] = "FORGED_HASH"
        if "phash" in metadata:
            metadata["phash"] = "FORGED_PHASH"
        with open(path, "w") as file:
            json.dump(metadata, file)

# ---------- Provenance verification ----------
def verify_provenance(image_path, use_phash=False, threshold=10):
    metadata_path = image_path + ".json"
    if not os.path.exists(metadata_path):
        return "missing"

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    if use_phash:
        original_ph = metadata.get("phash")
        if original_ph is None:
            return "missing"
        current_ph = compute_phash(image_path)
        dist = hamming_distance(original_ph, current_ph)
        return "valid" if dist <= threshold else "forged"
    else:
        # Original SHA256 method
        if metadata.get("hash") != compute_sha256(image_path):
            return "forged"
        return "valid"

def provenance_score(image_path):
    metadata_path = image_path + ".json"
    if not os.path.exists(metadata_path):
        return 0.5
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    if "phash" not in metadata:
        return 0.5
    dist = hamming_distance(metadata["phash"], compute_phash(image_path))
    return min(dist / 20.0, 1.0)

# ---------- Dataset splitting (used for Dataset #2) ----------
def create_splits_if_needed():
    """Create Train / Validation / Test splits if they don't exist."""
    if not (os.path.isdir(os.path.join(DATASET_PATH, "Train")) and
            os.path.isdir(os.path.join(DATASET_PATH, "Validation")) and
            os.path.isdir(os.path.join(DATASET_PATH, "Test"))):
        print("Creating train/val/test splits...")
        for split in ["Train", "Validation", "Test"]:
            for label in ["Fake", "Real"]:
                os.makedirs(os.path.join(DATASET_PATH, split, label), exist_ok=True)

        fake = [os.path.join(DATASET_PATH, "Fake", f) for f in os.listdir(os.path.join(DATASET_PATH, "Fake"))
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        real = [os.path.join(DATASET_PATH, "Real", f) for f in os.listdir(os.path.join(DATASET_PATH, "Real"))
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        fake_train, fake_temp = train_test_split(fake, test_size=0.3, random_state=42)
        fake_val, fake_test = train_test_split(fake_temp, test_size=0.5, random_state=42)

        real_train, real_temp = train_test_split(real, test_size=0.3, random_state=42)
        real_val, real_test = train_test_split(real_temp, test_size=0.5, random_state=42)

        for src_list, dst_label in [(fake_train, "Fake"), (real_train, "Real")]:
            for img in src_list:
                shutil.copy(img, os.path.join(DATASET_PATH, "Train", dst_label, os.path.basename(img)))
        for src_list, dst_label in [(fake_val, "Fake"), (real_val, "Real")]:
            for img in src_list:
                shutil.copy(img, os.path.join(DATASET_PATH, "Validation", dst_label, os.path.basename(img)))
        for src_list, dst_label in [(fake_test, "Fake"), (real_test, "Real")]:
            for img in src_list:
                shutil.copy(img, os.path.join(DATASET_PATH, "Test", dst_label, os.path.basename(img)))
        print("Splits created.")
    else:
        print("Using existing splits.")