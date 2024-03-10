import cv2
import numpy as np
import torch
from torchvision import transforms

# 接收模拟器的图片，转换成能直接输入给模型的tensor
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
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor

if __name__ == "__main__":
    img = cv2.imread('/home/shihong/桌面/Autonomous_Driving/Simulator/Linux/Data/Dataset10/IMG/CapturedImage3.jpg')
    output = preprocess_image_for_model(img)
    print(output.shape)