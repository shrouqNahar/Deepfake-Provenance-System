# model.py
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
import pandas as pd
import numpy as np
from config import *

def get_model():
    model = timm.create_model('efficientnet_b0', pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    return model.to(DEVICE)

def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)

def evaluate_model(model, test_dataset, test_loader, verifier, missing_penalty, forged_penalty):
    model.eval()
    all_probs = []
    all_labels = []
    all_preds = []
    all_paths = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            for j in range(len(probs)):
                idx = i * BATCH_SIZE + j
                image_path = test_dataset.samples[idx][0]
                status = verifier(image_path)
                score = probs[j].item()

                if status == "missing":
                    score = min(score + missing_penalty, 1.0)
                elif status == "forged":
                    score = min(score + forged_penalty, 1.0)

                pred = 1 if score > 0.5 else 0
                all_probs.append(score)
                all_labels.append(labels[j].item())
                all_preds.append(pred)
                all_paths.append(image_path)

    return all_labels, all_preds, all_probs, all_paths

def compute_metrics(all_labels, all_preds, all_probs):
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    report = classification_report(all_labels, all_preds, output_dict=True)
    return roc_auc, report, (fpr, tpr)