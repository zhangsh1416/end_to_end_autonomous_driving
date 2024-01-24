import torch

def convert_model_output_to_labels(throttle_brake_output, steering_output):
    # 对油门和刹车输出应用阈值
    throttle_label = (torch.sigmoid(throttle_brake_output[0]) >= 0.5).int()
    brake_label = (torch.sigmoid(throttle_brake_output[1]) >= 0.5).int()

    # 对转向角输出选择概率最高的类别
    _, steering_label_idx = torch.max(steering_output,0)
    steering_label = steering_label_idx - 1   # 将索引映射回原始标签

    return throttle_label, brake_label, steering_label
