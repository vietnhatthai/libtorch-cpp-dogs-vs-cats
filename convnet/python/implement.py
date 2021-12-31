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