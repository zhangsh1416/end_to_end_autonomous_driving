import socketio
# concurrent networking
# web server gateway interface
import eventlet.wsgi
from flask import Flask
import base64
import cv2
import numpy as np
# from io import BytesIO
import time
from outputs_to_labels import convert_model_output_to_labels
import torch
from torchvision import transforms
from networks.model_demo import create_custom_resnet_model


weights = torch.load('/home/shihong/桌面/Autonomous_Driving/autonomous_driving_simulator/train/end_to_end_{timestamp}.pt')
model = create_custom_resnet_model()
model.load_state_dict(weights)
model.eval()
# Initialize Socket.IO server
sio = socketio.Server()
app = Flask(__name__)

frame_count = 0
frame_count_save = 0
prev_time = 0
fps = 0
predicted_controls_global = None

def preprocess_image_for_model(img):
    # 将图像从BGR转换为RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将图像数据类型转换为float32，并归一化到[0, 1]
    img = img.astype(np.float32) / 255.0
    # 将numpy数组转换为torch张量
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)
    # 如果有可用的CUDA设备，则将张量转移到GPU上
    if torch.cuda.is_available():
        img_tensor = img_tensor.to('cuda')
    # 调整图像大小
    resize = transforms.Resize((224, 224), antialias=True)
    img_tensor = resize(img_tensor)
    return img_tensor

@sio.on("send_image")
def on_image(sid, data):
    # make the variables global to calculate the fps
    global frame_count, frame_count_save, prev_time, fps
    print("image recieved!")
    img_data = data["image"]
    img_bytes = base64.b64decode(img_data)
    # Decode image from base64 format，将字典里抽取出来的字符串转换为字节串类型
    img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

    # Calculate and print fps
    frame_count += 1
    elapsed_time = time.time() - prev_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        print(f"FPS: {fps:.2f}")
        prev_time = time.time()
        frame_count = 0

    # show the recieved images on the screen
    if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
        processed_img = preprocess_image_for_model(img)
        with torch.no_grad():
            throttle_brake_output, steering_output = model(processed_img.unsqueeze(0))
            throttle_brake_output = throttle_brake_output.squeeze(0)
            print(throttle_brake_output)
            steering_output = steering_output.squeeze(0)
            print(steering_output)
            # print(predicted_controls)
            predicted_controls = convert_model_output_to_labels(throttle_brake_output, steering_output)
        predicted_controls_global = predicted_controls
        print(predicted_controls_global)

        cv2.namedWindow("image from unity", cv2.WINDOW_NORMAL)
        cv2.imshow("image from unity", img)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            return
    else:
        print("Invalid image data")


# listen for the event "vehicle_data"
@sio.on("vehicle_data")
def vehicle_command(sid, data):
    global predicted_controls_global
    print("data recieved!")
    steering_angle = float(data["steering_angle"])
    throttle = float(data["throttle"])
    brake = float(data["brake"])
    velocity = float(data["velocity"])

    print(f"velocity of the car: {velocity}")

    if data:
        steering_angle = 0
        throttle = 1
        brake = 0
        if predicted_controls_global is not None:
            steering_angle, throttle, brake = predicted_controls_global
        send_control(steering_angle, throttle, brake)
    else:
        print("data is empty")


@sio.event
def connect(sid, environ):
    # sid for identifying the client connected表示客户端唯一标识符，environ表示其连接的相关环境信息
    print("Client connected")
    send_control(0, 0, 0)


# Define a data sending function to send processed data back to unity client
def send_control(steering_angle, throttle, brake):
    control_data = [str(throttle), str(brake), str(steering_angle)]
    sio.emit("control_command", data=control_data, skip_sid=True)


@sio.event
def disconnect(sid):
    # implement this function, if disconnected
    print("Client disconnected")


app = socketio.Middleware(sio, app)
# Connect to Socket.IO client
if __name__ == "__main__":
    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)