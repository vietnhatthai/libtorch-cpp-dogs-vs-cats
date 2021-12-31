import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from model import ConvNet
from dataset import Dataset

import numpy as np
import tqdm

# Device
if torch.cuda.is_available():
    print("CUDA available. Training on GPU.")
    device = torch.device("cuda")
else:
    print("Training on CPU.")
    device = torch.device("cpu")

# Hyperparameters
data_path = 'D:\\Projects\\pytorch-test\\data\\dogcat\\train'  # data root
CLASSES = ['dog', 'cat']
NUM_CLASSES = len(CLASSES)      # 2
INPUT_SIZE = 224                # 224 -w, 224 -h
BATCH_SIZE = 64
LEARING_RATE = 0.001
NUM_EPOCHS = 5
TESTSET_SIZE = 0.2