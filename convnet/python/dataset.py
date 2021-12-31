import torch
from torch.utils.data import Dataset
import glob
import os
import numpy as np 
from PIL import Image
from sklearn.preprocessing import LabelEncoder

class Dataset(Dataset):
    def __init__(self, path_root, classes, transform = None):
        super(Dataset, self).__init__()
        self.path_root = path_root
        self.classes = classes
        self.le = LabelEncoder()
        self.le.fit(self.classes)
        self.num_classes = len(classes)
        self.transform = transform
        self.listDatset = []
        self.__getlistitems()

    def __getlistitems(self):
        for image_path in glob.glob(os.path.join(self.path_root, "*.jpg")):
            self.listDatset.append(image_path.split('\\')[-1])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        name_image = self.listDatset[idx]
        label = name_image.split('.')[0]
        label = self.le.transform([label])[0]
        image = Image.open(os.path.join(self.path_root, name_image)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.listDatset)


if __name__ == '__main__':
    from torchvision import transforms
    from torch.utils.data import DataLoader

    root = 'D:\\Projects\\pytorch-test\\data\\dogcat\\train'
    classes = ['dog', 'cat']

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = Dataset(root, classes, transform)

    dataloader = DataLoader(dataset, batch_size=4,
                        shuffle=True, num_workers=0)
