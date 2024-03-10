import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data_preparation.end_to_end_data_pre import Dataset_unity  # 确保这个模块和类是可用的
import datetime


class EarlyStopping:
  """早停机制，防止过拟合"""

  def __init__(self, patience=7, min_delta=0):
    """
    :param patience: (int) 当验证集损失在连续几个epoch内未改善时，训练将停止
    :param min_delta: (float) 表示被认为是改善的最小变化
    """
    self.patience = patience
    self.min_delta = min_delta
    self.counter = 0
    self.best_loss = None
    self.early_stop = False

  def __call__(self, val_loss):
    if self.best_loss is None:
      self.best_loss = val_loss
    elif val_loss > self.best_loss - self.min_delta:
      self.counter += 1
      if self.counter >= self.patience:
        self.early_stop = True
    else:
      self.best_loss = val_loss
      self.counter = 0
def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # 创建数据集
  path_train = '../data/train_set.csv'
  path_val = '../data/test_set.csv'
  train_data = Dataset_unity(data_path=path_train)
  val_data = Dataset_unity(data_path=path_val)

  # 加载数据集
  BATCH_SIZE = 128

  
  trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
  testloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)  # 验证集一般不需要shuffle

  # 定义模型
  resnet = models.resnet50(pretrained=True)
  for param in resnet.parameters():
    param.requires_grad = False
  num_features = resnet.fc.in_features
  resnet.fc = nn.Sequential(nn.Linear(num_features, 1), nn.Tanh())
  resnet.to(device)

  # 定义损失函数和优化器
  criterion = nn.MSELoss()
  optimizer = optim.Adam(resnet.fc.parameters(), lr=0.0005)

  # 训练和验证
  EPOCHS = 100
  train_losses = []
  val_losses = []

  print("=========================================== START TRAINING ===========================================")
  early_stopping = EarlyStopping(patience=10, min_delta=0.001)
  for epoch in range(EPOCHS):
    resnet.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
      inputs, labels = inputs.to(device), labels.view(-1, 1).to(device)  # 确保labels是正确的形状

      optimizer.zero_grad()
      outputs = resnet(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

    # 计算平均训练损失
    train_loss = running_loss / len(trainloader)
    train_losses.append(train_loss)

    # 验证过程
    resnet.eval()
    val_running_loss = 0.0
    with torch.no_grad():
      for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.view(-1, 1).to(device)
        outputs = resnet(inputs)
        val_loss = criterion(outputs, labels)
        val_running_loss += val_loss.item()

    # 计算平均验证损失
    val_loss = val_running_loss / len(testloader)
    val_losses.append(val_loss)

    early_stopping(val_loss)
    if early_stopping.early_stop:
      print("早停机制启动，终止训练！")
      break

    print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

  # 结果可视化
  plt.plot(train_losses, label='Training Loss')
  plt.plot(val_losses, label='Validation Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Training and Validation Loss')
  plt.legend()
  plt.show()

  # 保存模型
  now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  filename = f"model_resnet50_{now}.pth"
  torch.save(resnet.state_dict(), filename)



if __name__ == "__main__":
  main()
