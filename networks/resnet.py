from torch import nn
from torchvision import models


#定义模型 define model
resnet = models.resnet50(pretrained=True)

#冻结所有权重 freeze all trainable parameters
for param in resnet.parameters():
  param.requires_grad = False
#替换Resnet50的全连接层 replace the FC layer of Resnet50
num_features =  resnet.fc.in_features
resnet.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=3), nn.Tanh())



































"""
from torch import nn
from torchvision import models

def resnet_model(weights='ResNet50_Weights.DEFAULT'):
    model = models.resnet50(weights = weights)
    model.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=3), nn.Tanh())
    return model

# 测试网络
if __name__ == '__main__':
    model = resnet_model()
    #print(model)
"""