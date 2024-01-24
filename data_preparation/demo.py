import cv2
import os
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('TkAgg')
from end_to_end_data_pre import Dataset_unity
from torchvision import models
from torch import nn
#检查cuda是否可用
print(torch.cuda.is_available())
#初始化模型
resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
resnet.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=3), nn.Tanh())
if torch.cuda.is_available():
    resnet = resnet.to('cuda')
#print(resnet)
#创建数据集
path_train = '/home/shihong/桌面/Autonomous_Driving/Simulator/Linux/Data/Dataset10/VehicleData.txt'
path_val = '/home/shihong/桌面/Autonomous_Driving/Simulator/Linux/Data/Dataset11/VehicleData.txt'
train_data = Dataset_unity(data_path=path_train)
val_data = Dataset_unity(data_path=path_val)
#创建迭代器
train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=1, pin_memory=True)
val_dataloader = DataLoader(val_data, batch_size=4, shuffle=False, num_workers=1, pin_memory=True)
#设置优化器，损失函数
optimizer = torch.optim.SGD(resnet.parameters(), lr=0.001)
loss_function = torch.nn.MSELoss()
#训练循环，每个epoch运行一次val数据集
for i in range(10):
    resnet.train()
    for idx, (input,lables) in enumerate(train_dataloader):
        input, lables = input.to('cuda'), lables.to('cuda')
        #print(input.size())
        #print(input.device)
        optimizer.zero_grad()
        output = resnet(input)
        loss = loss_function(output, lables)
        #print(loss.item())
        loss.backward()
        optimizer.step()
        # 训练完每个epoch进行验证
    resnet.eval()
    with torch.no_grad():
        loss_sum = 0
        for val_i, (input, target) in enumerate(val_dataloader):
            input, target = input.to('cuda'), target.to('cuda')
            output = resnet(input)
            loss_sum = loss_sum + loss_function(output, target)
        print(val_i)
        print('val_loss:', loss_sum.item() / (val_i + 1))

torch.save(resnet, '/home/shihong/桌面/Autonomous_Driving/Simulator/Linux/Data/Dataset10' + '/' + str(i) + '.pt')
torch.cuda.empty_cache()


#print(input.size(), lables.size())































"""
os.chdir('/home/shihong/桌面/Autonomous_Driving')
print(os.getcwd())


data_path = './Simulator/Linux/Data/Dataset10/IMG/CapturedImage0.jpg'
image = cv2.imread(data_path)
image.resize(224,224,3)
print(image.shape)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('my picture')
plt.show()


#image = Image.open('/home/shihong/桌面/Autonomous_Driving/Simulator/Linux/Data/Dataset10/IMG/CapturedImage0.jpg')
#image_resized = image.resize((224,224))
#print(image.shape)
# 显示图像
#image.show()
#print(image_resized.size)
"""
