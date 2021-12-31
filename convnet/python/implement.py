import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder

from model import ConvNet

import os
import numpy as np
import cv2
import glob
import tqdm

# Device
if torch.cuda.is_available():
    print("CUDA available. Training on GPU.")
    device = torch.device("cuda")
else:
    print("Training on CPU.")
    device = torch.device("cpu")

# Hyperparameters
test_path = 'D:\\Projects\\pytorch-test\\data\\dogcat\\test1'  # test root
CLASSES = ['dog', 'cat']
NUM_CLASSES = len(CLASSES)      # 2
INPUT_SIZE = 224                # 224 -w, 224 -h

# LabelEncoder
le = LabelEncoder()
le.fit(CLASSES)