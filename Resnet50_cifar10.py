import torch
import torchvision
from torchvision import datasets, transforms, models
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#对图像进行预处理，调节像素尺寸为224x224,将图片数据转换成Tensor，进行归一化处理。
#preprocess the images, scale the image size to 224x224, transform image to tensor, normalization.
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#创建数据集 creat dataset
trainset = datasets.CIFAR10(root="./data/cifar-10-python",
                            download=False,
                            train=True,
                            transform=transform)

testset = datasets.CIFAR10(root="./data/cifar-10-python",
                           download=False,
                           train=False,
                           transform=transform)
#加载数据 load dataset
BATCH_SIZE = 128
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=2)
testloader = torch.utils.data.DataLoader(testset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=2)

#定义模型 define model
resnet = models.resnet50(pretrained=True)

#冻结所有权重 freeze all trainable parameters
for param in resnet.parameters():
  param.requires_grad = False
#替换Resnet50的全连接层 replace the FC layer of Resnet50
num_features =  resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 10)

#将模型转到GPU move model to GPU
resnet.to(device)

#定义损失函数和优化器 define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=resnet.fc.parameters(), lr=0.005, momentum=0.9)

#开始训练 start training
EPOCH = 100
print("=========================================== START TRAINING ===========================================")
for epoch in range(EPOCH):
  running_loss = 0.0
  correct, total = 0, 0
  for i, data in enumerate(trainloader, 0):
    # Load images and labels
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)

    # Reset the gradient to zero
    optimizer.zero_grad()

    # Forward pass
    outputs = resnet(inputs)
    # Calculate loss
    loss = criterion(outputs, labels)
    # Backward pass
    loss.backward()
    # Update the weights
    optimizer.step()

    # Calculate loss for each iteration
    running_loss += loss.item()
    # Get the predictions
    _, predicted = torch.max(outputs.data, 1)
    # Update total and correct predictions
    total += labels.size(0)
    correct += (predicted==labels).sum().item()

    # Print loss and training accuracy for every 100th iteration for each epoch
    if i%100 == 99:
      print(f"epoch: {epoch+1}, running_loss: {running_loss/100}, Training Accuracy: {correct/total*100}")
      running_loss = 0.0
print("=========================================== FINISHED TRAINING ==========================================")

#保存模型
PATH = "./resnet_model.pt"
torch.save(resnet, PATH)

def accuracy_calculation(testloader, model, attack=False):
  correct, total = 0, 0
  for data in testloader:
    # Extract input images and respective labels
    inputs, labels = data
    # Move the images and their labels to device (for GPU processing)
    inputs, labels = inputs.to(device), labels.to(device)
    # This condition checks whether the input images are adversarial or not
    # The model consumes clean test images
    outputs = resnet(inputs)
    # Get the predictions
    _, predicted = torch.max(outputs.data, 1)

    # Update total and correct predictions and return them
    total += labels.size(0)
    correct += (predicted==labels).sum().item()
  return correct, total

test_accuracy = correct/total
print("test_accurcy:",test_accuracy)