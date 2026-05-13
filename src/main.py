# main.py
import os
import shutil
import kagglehub
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from config import *
from utils import (
    generate_metadata_for_folder, remove_random_metadata, forge_random_metadata,
    verify_provenance, create_splits_if_needed
)
from model import get_model, train_one_epoch, evaluate_model, compute_metrics
from pipeline import (
    evaluate_provenance_survivability, ablation_study, computational_overhead
)

def download_and_prepare_dataset(dataset_name, split_exists=True):
    """
    Download dataset from Kaggle, copy to DATASET_PATH, create metadata,
    simulate provenance attacks.
    """
    path = kagglehub.dataset_download(dataset_name)
    print(f"Dataset downloaded to {path}")

    if os.path.exists(DATASET_PATH):
        shutil.rmtree(DATASET_PATH)
    shutil.copytree(path, DATASET_PATH)

    # For dataset that already has splits (dataset #1)
    if not split_exists:
        create_splits_if_needed()  # Dataset #2 case

    # Generate metadata
    for split in ["Train", "Validation", "Test"]:
        for label in ["Fake", "Real"]:
            folder = os.path.join(DATASET_PATH, split, label)
            if os.path.exists(folder):
                generate_metadata_for_folder(folder)

    # Simulate provenance attacks on Test set
    for label in ["Fake", "Real"]:
        folder = os.path.join(DATASET_PATH, "Test", label)
        remove_random_metadata(folder, 0.2)
        forge_random_metadata(folder, 0.1)

    print("Dataset prepared, metadata generated, attacks simulated.")

def run_full_pipeline():
    # 1. Prepare dataset (choose one; here we use Dataset #2 which requires split creation)
    download_and_prepare_dataset("kshitizbhargava/deepfake-face-images", split_exists=False)

    # 2. Create DataLoaders
    train_dataset = datasets.ImageFolder(os.path.join(DATASET_PATH, "Train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATASET_PATH, "Validation"), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(DATASET_PATH, "Test"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Model setup
    model = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training
    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        print(f"Epoch {epoch+1} loss: {loss:.4f}")

    # 5. Basic evaluation with hybrid adjustment
    all_labels, all_preds, all_probs, all_paths = evaluate_model(
        model, test_dataset, test_loader,
        lambda p: verify_provenance(p), MISSING_PENALTY, FORGED_PENALTY
    )
    roc_auc, report, (fpr, tpr) = compute_metrics(all_labels, all_preds, all_probs)
    print(f"\nHybrid AUC: {roc_auc:.4f}")

    # 6. Advanced analysis
    surv_df = evaluate_provenance_survivability(test_dataset)
    ablation_df = ablation_study(model, test_dataset, test_loader)
    overhead = computational_overhead(model, test_loader)

    # 7. Save outputs
    surv_df.to_csv("provenance_survivability.csv", index=False)
    ablation_df.to_csv("ablation_results.csv", index=False)
    print("All results saved.")

if __name__ == "__main__":
    run_full_pipeline()