import torch
import torchvision.models as models
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet = models.resnet50(weights=True)
for param in resnet.parameters():
    param.requires_grad = False
num_features = resnet.fc.in_features
resnet.fc = nn.Sequential(nn.Linear(num_features, 1), nn.Tanh())
resnet.to(device)
