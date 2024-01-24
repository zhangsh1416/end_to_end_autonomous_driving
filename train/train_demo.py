import torch
import torchvision
from torchvision import datasets, transforms, models
import torch.nn as nn
from autonomous_driving_simulator.data_preparation.end_to_end_data_pre import Dataset_unity
from autonomous_driving_simulator.networks.model_demo import create_custom_resnet_model
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
import torch.optim as optim
def train_and_evaluate_model(trainloader, testloader, device, model, optimizer, epochs,patience=10, min_delta=0.001):
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    epochs_no_improve = 0
    print("=========================================== START TRAINING ===========================================")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            # print(labels.shape)
            throttle_labels = labels[:, 0]
            brake_labels = labels[:, 1]
            steering_labels = labels[:, 2]
            optimizer.zero_grad()
            throttle_brake_output, steering_output = model(inputs)
            throttle_output, brake_output = throttle_brake_output[:, 0], throttle_brake_output[:, 1]
            # print(throttle_output[5])
            # print(labels)
            loss_throttle = F.binary_cross_entropy_with_logits(throttle_output, throttle_labels)
            loss_brake = F.binary_cross_entropy_with_logits(brake_output, brake_labels)
            steering_labels = steering_labels.long()
            # cross entropy要求输入值不能是负数，因此将labels整体加1,在推理阶段会对应的减出来
            steering_labels += 1
            loss_steering = F.cross_entropy(steering_output, steering_labels)
            loss = loss_throttle + loss_brake + loss_steering
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f"Epoch {epoch+1}, Batch {i+1}/{len(trainloader)}, Batch Loss: {loss.item()}")

        # 在每个 epoch 结束时计算平均训练损失
        train_loss = running_loss / len(trainloader)
        train_losses.append(train_loss)

        # 评估模型
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                throttle_labels = labels[:, 0]
                brake_labels = labels[:, 1]
                steering_labels = labels[:, 2]
                throttle_brake_output, steering_output = model(inputs)
                throttle_output, brake_output = throttle_brake_output[:, 0], throttle_brake_output[:, 1]
                # print(labels)
                loss_throttle = F.binary_cross_entropy_with_logits(throttle_output, throttle_labels)
                loss_brake = F.binary_cross_entropy_with_logits(brake_output, brake_labels)
                steering_labels = steering_labels.long()
                steering_labels += 1
                loss_steering = F.cross_entropy(steering_output, steering_labels)
                loss = loss_throttle + loss_brake + loss_steering
                val_loss += loss.item()

        # 在每个 epoch 结束时计算平均验证损失
        val_loss = val_loss / len(testloader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss}")

        if (best_val_loss - val_loss) > min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss}, Best Val Loss: {best_val_loss}")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    print(
            "=========================================== FINISHED TRAINING ==========================================")

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建数据集
    path_train = '/home/shihong/桌面/Autonomous_Driving/Simulator/Linux/Data/Dataset24/VehicleData.txt'
    path_val = '/home/shihong/桌面/Autonomous_Driving/Simulator/Linux/Data/Dataset11/VehicleData.txt'
    train_data = Dataset_unity(data_path=path_train)
    val_data = Dataset_unity(data_path=path_val)

    # 加载数据
    BATCH_SIZE = 128
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # 定义模型
    custom_model = create_custom_resnet_model()

    # 定义优化器
    optimizer = optim.SGD([
        {'params': custom_model.resnet.parameters(), 'lr': 0.001},  # 对预训练层使用较小的学习率
        {'params': custom_model.throttle_brake_layer.parameters(), 'lr': 0.01},  # 对油门刹车层使用较大的学习率
        {'params': custom_model.steering_layer.parameters(), 'lr': 0.01}  # 对转向角层使用较大的学习率
    ], momentum=0.9)

    # 训练和评估模型
    EPOCHS = 100
    train_and_evaluate_model(trainloader, testloader, device, custom_model, optimizer, EPOCHS)

    # 保存模型
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    PATH = './end_to_end_{timestamp}.pt'
    torch.save(custom_model.state_dict(), PATH)


if __name__ == "__main__":
    main()

