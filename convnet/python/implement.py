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

# Transforms
transform = transforms.Compose([
    transforms.ToPILImage(),            # convert CV2 to PIL
    transforms.ToTensor(),
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Neural Network model
model = ConvNet(NUM_CLASSES)
model.load_state_dict(torch.load('model.pth'))
model.to(device)
model.eval()
print(model)

# Prediction data test
print("[INFO] Testing model...")
for image_name in glob.glob(os.path.join(test_path, "*.jpg")):
    image = cv2.imread(image_name)
    image_tensor = transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).to(device)
    image_tensor = image_tensor.unsqueeze(0)

    outputs = model(image_tensor)
    prediction = np.argmax(outputs.cpu().detach().numpy(), 1)[0]
    prediction = le.classes_[prediction]
    print("[TEST] Prediction : ", prediction)

    cv2.putText(image, prediction, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("image", image)
    if cv2.waitKey(1) & ord('q')==0xff:
        break

# Close
cv2.destroyAllWindows()
cv2.waitKey(1)