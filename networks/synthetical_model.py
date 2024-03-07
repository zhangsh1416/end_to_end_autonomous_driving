# 实现resnet和yolo的综合控制，通过synthetical_model实现有输入的图片unmpy数组（由cv2.imread得到），输出最终控制，完成主要逻辑
import torch
from networks.end_to_end_model import resnet  # 确保导入了模型定义
import cv2
import numpy as np
import torchvision.transforms.functional as TF
from ultralytics import YOLO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载模型并将其移动到正确的设备
model = resnet.to(device)
model.load_state_dict(torch.load('../train/model_resnet50_20240306-221138.pth', map_location=device))
model.eval()
yolo = YOLO('yolov8n.pt')
# 输入cv2读取的图片numpy数组，输出可以直接feed进model的tensor,采用的是opencv接口。
def preprocess_image_for_model(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)
    img_tensor = TF.resize(img_tensor, (224, 224))
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)
    return img_tensor
# 输入cv2读取的图片numpy数组，取中间图片，输出同样可以直接feed进yolov8，yolov8拥有非常强大的接口，支持大多数图片数据类型作为输入。
def preprocess_for_yolov8(img):
    img_yolov8 = img[:, int(img.shape[1] / 3):int(img.shape[1] * 2 / 3), :]
    return img_yolov8
# 根据yolov8的检测结果判断前方是否有车辆，输入有bb（xywhn），和两个阈值（超参），两个阈值分别是中心偏离阈值和bb大小阈值。
def is_vehicle_in_front(bbox, threshold_x, threshold_y):
    # 假设bbox为[1, 5]形状的Tensor，其中包含单个边界框的xywhn
    # 转换bbox以匹配中心点计算
    center_x, center_y = bbox[0, 0], bbox[0, 1]

    # 计算中心点偏差
    center_diff_x = torch.abs(0.5 - center_x)
    center_diff_y = torch.abs(0.5 - center_y)

    # 获取宽度和高度
    bbox_width = bbox[0, 2]
    bbox_height = bbox[0, 3]

    # 判断逻辑
    if center_diff_x < threshold_x and center_diff_y < threshold_x and bbox_width > threshold_y and bbox_height > threshold_y:
        return True
    else:
        return False

# 根据yolov8的检测结果判断是否有红灯，输入检测到的交通灯bb（xyxy）和原图像numpy数组。
def is_red_traffic_light_detected(xyxy,img):
    bbox = xyxy
    # 裁剪交通灯区域
    traffic_light = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    # 将RGB图像转换到HSV色彩空间
    hsv = cv2.cvtColor(traffic_light, cv2.COLOR_BGR2HSV)

    # 定义红色的HSV阈值
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # 根据阈值构建掩模，然后进行位运算
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    full_mask = mask1 + mask2
    red_result = cv2.bitwise_and(traffic_light, traffic_light, mask=full_mask)

    # 检查红色区域是否占据显著部分
    # 这里简化为检查掩模中非零像素的比例
    if np.count_nonzero(full_mask) / (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) > 0.2:  # 假设阈值
        return True
    else:
        return False
def synthetical_model(img):
    brake = 0
    throttle = 1
    steering_angle = 0
    img_for_yolo = preprocess_for_yolov8(img)
    vehicle_results = yolo.predict(source=[img_for_yolo], classes=[2,3,5,7])
# 判断前方车辆
    for i in vehicle_results:
        vehicle_bb = i.boxes.xywhn
        if is_vehicle_in_front(vehicle_bb,0.1,0.1):
            brake = 1
# 检测交通灯
    if brake == 0:  # 只有当未因车辆刹车时才检查交通灯
        traffic_light_results = yolo.predict(source=[img_for_yolo], classes=[9])
        for j in traffic_light_results:
            tl_bb = j.boxes.xyxy
            if is_red_traffic_light_detected(tl_bb, img_for_yolo):
                brake = 1
                break  # 如果检测到红灯，设置刹车为1并停止检查其余的交通灯
    throttle = 0 if brake == 1 else 1
    img_for_model = preprocess_image_for_model(img_for_yolo)
    steering_angle = model(img_for_model)
    return throttle,brake,steering_angle

if __name__ == "__main__":
    img = cv2.imread('../data/IMG/CapturedImage466.jpg')
    x,y,z = synthetical_model(img)
    print(x,y,z)






