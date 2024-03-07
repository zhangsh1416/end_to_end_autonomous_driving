# 测试模型是否可以成功加载
import torch
import torch.nn as nn
from networks.model_demo import get_pretrained_resnet
from networks.model_demo import create_custom_resnet_model
import cv2
import numpy as np
from datetime import datetime
from torchvision import transforms

pretrained_weights = torch.load('train/end_to_end_{timestamp}.pt')

model = create_custom_resnet_model()
model.eval()
model.load_state_dict(pretrained_weights)
# print(model)

def preprocess_image_for_model(img):
    # 将图像从BGR转换为RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将图像数据类型转换为float32，并归一化到[0, 1]
    img = img.astype(np.float32) / 255.0
    # 将numpy数组转换为torch张量
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)
    # 如果有可用的CUDA设备，则将张量转移到GPU上
    if torch.cuda.is_available():
        img_tensor = img_tensor.to('cuda')
    # 调整图像大小
    resize = transforms.Resize((224, 224), antialias=True)
    img_tensor = resize(img_tensor)
    # print(img_tensor.device)
    img_tensor = img_tensor.unsqueeze(0)  # 这会将其变形为[1, 3, 224, 224]

    return img_tensor


img = cv2.imread('CapturedImage199.jpg')

img_tensor = preprocess_image_for_model(img)
start_time = datetime.now()
output = model(img_tensor)
end_time = datetime.now()
elapsed_time = (end_time - start_time).total_seconds()
print(output)
print(f"Model ran in {elapsed_time} seconds.")