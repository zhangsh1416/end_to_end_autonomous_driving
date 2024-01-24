from torch import nn
from torchvision import models
import torch

def get_pretrained_resnet(device='cuda'):
    resnet = models.resnet50(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad = False
    num_features = resnet.fc.in_features
    if device:
        resnet = resnet.to(device)
    return resnet, num_features

def create_custom_resnet_model(device='cuda'):
    original_resnet, num_features = get_pretrained_resnet(device=device)
    custom_model = CustomResNet(original_resnet, num_features).to(device)
    return custom_model

class CustomResNet(nn.Module):
    def __init__(self, original_resnet, num_features):
        super(CustomResNet, self).__init__()
        self.resnet = nn.Sequential(*(list(original_resnet.children())[:-1]))
        self.throttle_brake_layer = nn.Linear(num_features, 2)
        self.steering_layer = nn.Linear(num_features, 3)

    def forward(self, x):
        x = self.resnet(x).flatten(1)
        throttle_brake_output = self.throttle_brake_layer(x)
        steering_output = self.steering_layer(x)
        return throttle_brake_output, steering_output


if __name__ == "__main__":
    # 创建模型实例
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    custom_model = create_custom_resnet_model(device=device)
    print(custom_model)












"""
from torch import nn
from torchvision import models

# 这里最后fc层的改动是多余的，但是为了保留我的修改过程，就保留了下来
def get_pretrained_resnet(num_classes=1, device='cuda'):
    resnet = models.resnet50(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad = False
    num_features = resnet.fc.in_features
    resnet.fc = nn.Sequential(nn.Linear(in_features=num_features, out_features=num_classes), nn.Tanh())

    if device:
        resnet = resnet.to(device)

    return resnet


class CustomResNet(nn.Module):
    def __init__(self, original_resnet, num_features):
        super(CustomResNet, self).__init__()
        self.resnet = original_resnet
        self.throttle_brake_layer = nn.Linear(num_features, 2)
        self.steering_layer = nn.Linear(num_features, 3)

    def forward(self, x):
        x = self.resnet(x)
        throttle_brake_output = self.throttle_brake_layer(x)
        steering_output = self.steering_layer(x)
        return throttle_brake_output, steering_output

original_resnet = get_pretrained_resnet()
num_features = 10

# 替换原有的fc层
original_resnet.fc = nn.Identity()  # 使用nn.Identity()作为占位符

# 创建自定义模型
custom_model = CustomResNet(original_resnet, num_features)

# print(custom_model)

"""


