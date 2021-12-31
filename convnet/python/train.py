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