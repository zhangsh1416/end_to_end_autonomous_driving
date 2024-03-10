import cv2
import torch
from torch.utils.data import Dataset
import os
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

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
        return len(self.data)
    def __getitem__(self, item):
        self.data[item].strip()
        #print(self.root)
        #print(self.data[:3])
        parts = self.data[item].strip().split(',')
        img_name, throttle, brake, steering, velocity = parts
        if not img_name.startswith('/'):
            img_name = '/' + img_name
        if "IMG" in img_name and img_name.count('/') < 2:  # 如果路径中"IMG"后面缺少斜杠
            img_name = img_name.replace("IMG", "IMG/")  # 添加缺失的斜杠
        img_abspath = os.path.join(self.root, img_name.lstrip('/'))  # 构造绝对路径
        img = cv2.imread(img_abspath)

        if img is None:
            raise FileNotFoundError(f"Image {img_abspath} not found.")
        #print(type(img_name))
        #print(img_abspath)
        img = img[:, int(img.shape[1] / 3):int(img.shape[1] * 2 / 3), :]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img= img.astype(np.float32) / 255.0
        img = torch.from_numpy(img)
        label = torch.from_numpy(np.array([steering]).astype(np.float32))
        label = label.float()
        # label = torch.tensor([float(steering)], dtype=torch.float32)
        #print(f"Requesting index: {item}, Data length: {len(self.data)}")
        #print(img.size())
        img = img.permute(2,0,1)
        resize = transforms.Resize((224, 224), antialias=True)
        img = resize(img)
        #print(img.shape)
        #print(label)
        #print(img.size())
        return img, label
if __name__ == "__main__":
    x = Dataset_unity('../data/train_set.csv')
    x.__len__()
    print(len(x))
    x.__getitem__(3419)
    #print(x.__getitem__(3419))

    y = Dataset_unity('../data/test_set.csv')
    print(len(y))
    print(y.__getitem__(855))
