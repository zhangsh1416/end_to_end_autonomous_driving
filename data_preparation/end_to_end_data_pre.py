import cv2
import torch
from torch.utils.data import Dataset
import os
import numpy as np
from torchvision import transforms

class Dataset_unity(Dataset):
    def __init__(self, data_path):
        self.root = os.path.dirname(data_path)
        #print(self.root)
        f = open(data_path, 'r')
        #print(type(f))
        with open(data_path, 'r') as f:
            self.data = f.readlines()[1:]  # 直接跳过第一行

    def __len__(self):
        #print(len(self.data)-1)
        return len(self.data)-1
    def __getitem__(self, item):
        self.data[item].strip()

        #print(self.root)
        #print(self.data[:3])
        parts = self.data[item].split(' ')
        #print(len(parts))
        img_name, stop, throttle, steering, velocity, position_x, position_y, space = parts
        img_name = img_name.lstrip('/')
        #print(type(img_name))
        img_abspath = os.path.join(self.root, img_name)
        #print(img_abspath)
        img = cv2.imread(img_abspath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img= img.astype(np.float32) / 255.0
        img = torch.from_numpy(img)
        label = torch.from_numpy(np.array([stop,throttle,steering]).astype(np.float32))
        label = label.float()
        # label = torch.tensor([float(steering)], dtype=torch.float32)
        #print(f"Requesting index: {item}, Data length: {len(self.data)}")
        #print(img.size())
        img = img.permute(2,0,1)
        resize = transforms.Resize((224, 224), antialias=True)
        img = resize(img)
        #print(label)
        #print(img.size())

        return img, label

x = Dataset_unity('/home/shihong/桌面/Autonomous_Driving/Simulator/Linux/Data/Dataset10/VehicleData.txt')
x.__len__()
x.__getitem__(1)
