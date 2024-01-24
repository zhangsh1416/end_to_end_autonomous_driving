import torch
import torchvision
from torchvision import datasets, transforms, models
import torch.nn as nn
from autonomous_driving_simulator.data_preparation.end_to_end_data_pre import Dataset_unity


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#创建数据集 creat dataset
path_train = '/home/shihong/桌面/Autonomous_Driving/Simulator/Linux/Data/Dataset24/VehicleData.txt'
path_val = '/home/shihong/桌面/Autonomous_Driving/Simulator/Linux/Data/Dataset11/VehicleData.txt'
train_data = Dataset_unity(data_path=path_train)
val_data = Dataset_unity(data_path=path_val)
#加载数据 load dataset
BATCH_SIZE = 128
trainloader = torch.utils.data.DataLoader(train_data,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=2)
testloader = torch.utils.data.DataLoader(val_data,
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
resnet.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=3), nn.Tanh())

#将模型转到GPU move model to GPU
resnet.to(device)

#定义损失函数和优化器 define loss function and optimizer
criterion = nn.MSELoss()
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
    # print(outputs.shape)
    # Calculate loss
    loss = criterion(outputs, labels)
    # Backward pass
    loss.backward()
    # Update the weights
    optimizer.step()

    # Calculate loss for each iteration
    running_loss += loss.item()
    # Update total and correct predictions
    total += labels.size(0)
    print(f"total:{total}")
    # print(outputs.shape)
    # print(labels.shape)
    correct_predictions = torch.all(outputs == labels, dim=1)
    correct += correct_predictions.sum().item()
    #print((outputs==labels).sum().item())
    print(f"correct:{correct}")

    # Print loss and training accuracy for every 100th iteration for each epoch
  print(f"epoch: {epoch+1}, running_loss: {running_loss/100}, Training Accuracy: {correct/total*100}")
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

    # Update total and correct predictions and return them
    total += labels.size(0)
    correct += (outputs==labels).sum().item()
  return correct, total

test_accuracy = correct/total
print("test_accurcy:",test_accuracy)