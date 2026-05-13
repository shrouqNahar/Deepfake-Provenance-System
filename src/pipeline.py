# pipeline.py
import os
import random
import time
import io
import shutil
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter
import imagehash
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import roc_curve, auc
import torch
from config import *
from utils import provenance_score, verify_provenance, compute_phash, hamming_distance
from model import get_model, evaluate_model

def evaluate_provenance_survivability(test_dataset, n_transforms=3, runs=5):
    print(f"\nEvaluating provenance survivability ({n_transforms} transforms, {runs} runs)...")
    results = []
    for run in range(runs):
        survived = 0
        total = 0
        for i, (path, _) in enumerate(test_dataset.samples):
            if verify_provenance(path, use_phash=True) == "missing":
                continue
            total += 1
            temp_img = os.path.join(TEMP_PATH, f"temp_{run}_{i}.jpg")
            temp_json = temp_img + ".json"
            shutil.copy(path, temp_img)
            shutil.copy(path + ".json", temp_json)
            img = Image.open(temp_img).convert("RGB")
            for _ in range(n_transforms):
                # Social IoT transformations
                if random.random() > 0.4:
                    img = transforms.functional.resize(img, random.choice([512, 720, 1080]))
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=random.randint(70, 90))
                img = Image.open(buffer)
                if random.random() > 0.5:
                    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.6)))
                # Randomly remove or forge metadata
                if random.random() < 0.3:
                    if os.path.exists(temp_json):
                        os.remove(temp_json)
                elif random.random() < 0.5:
                    fake_hash = str(imagehash.phash(img))
                    with open(temp_json, "w") as f:
                        json.dump({"phash": fake_hash}, f)
            img.save(temp_img)
            if verify_provenance(temp_img, use_phash=True) == "valid":
                survived += 1
        rate = (survived / total * 100) if total > 0 else 0
        results.append({'run': run, 'survival_rate_%': round(rate, 2), 'n_transforms': n_transforms})
    df = pd.DataFrame(results)
    print(df)
    return df

def ablation_study(model, test_dataset, test_loader):
    print("\nRunning Ablation Study on Penalty Values...")
    results = []
    for miss in [0.05, 0.1, 0.15, 0.2, 0.25]:
        for forge in [0.25, 0.3, 0.35, 0.4, 0.45]:
            all_labels, _, all_probs, _ = evaluate_model(
                model, test_dataset, test_loader,
                lambda p: verify_provenance(p), miss, forge)
            # Also get detection-only scores (no adjustment)
            _, _, det_probs, _ = evaluate_model(
                model, test_dataset, test_loader,
                lambda p: "valid", 0.0, 0.0)  # no penalty
            fpr_d, tpr_d, _ = roc_curve(all_labels, det_probs)
            fpr_h, tpr_h, _ = roc_curve(all_labels, all_probs)
            auc_d = auc(fpr_d, tpr_d)
            auc_h = auc(fpr_h, tpr_h)
            results.append({
                'penalty_missing': miss,
                'penalty_forged': forge,
                'AUC_detection': auc_d,
                'AUC_hybrid': auc_h,
                'Gain': auc_h - auc_d
            })
            print(f"miss={miss}, forge={forge}, AUC_det={auc_d:.4f}, AUC_hyb={auc_h:.4f}, Gain={auc_h-auc_d:.4f}")
    df = pd.DataFrame(results)
    return df

def computational_overhead(model, test_loader):
    print("\nComputational Overhead Analysis")
    times = []
    dummy = next(iter(test_loader))[0].to(DEVICE)
    for _ in range(30):
        start = time.time()
        _ = model(dummy)
        times.append(time.time() - start)
    avg_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Inference: {avg_ms:.2f} ms ± {std_ms:.2f}, Params: {params:.2f}M")
    return avg_ms, std_ms, params