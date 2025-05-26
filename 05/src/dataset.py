import numpy as np
import torch
from torchvision import transforms
from PIL import Image

class train_dataset(torch.utils.data.Dataset):
    def __init__(self, x_train, t_train, transform=None):
        data = x_train.astype('float32')
        self.x_train = [Image.fromarray(np.uint8(img)) for img in data]
        self.t_train = t_train
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        x = self.x_train[idx]
        if self.transform:
            x = self.transform(x)
        t = torch.tensor(self.t_train[idx], dtype=torch.long)
        return x, t

class test_dataset(torch.utils.data.Dataset):
    def __init__(self, x_test, transform=None):
        data = x_test.astype('float32')
        self.x_test = [Image.fromarray(np.uint8(img)) for img in data]
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.x_test)

    def __getitem__(self, idx):
        x = self.x_test[idx]
        if self.transform:
            x = self.transform(x)
        return x
