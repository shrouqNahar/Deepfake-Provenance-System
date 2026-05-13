# config.py
import torch
from torchvision import transforms

# Paths
DATASET_PATH = "/content/Dataset"
TEMP_PATH = "/content/temp_eval"

# Image processing
IMG_SIZE = 224
BATCH_SIZE = 2
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# Training
EPOCHS = 6
LEARNING_RATE = 1e-4

# Provenance penalties (default)
MISSING_PENALTY = 0.15
FORGED_PENALTY = 0.35
