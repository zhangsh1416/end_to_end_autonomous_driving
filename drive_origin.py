import socketio
import eventlet
import eventlet.wsgi
from flask import Flask
import base64
import cv2
import numpy as np
import random

# Create an instance of a WebSocket server based on Flask and Socket.IO.
sio_server = socketio.Server()
app = Flask(__name__)


# Listen for incoming connections
@sio_server.event
def connect(sid, environ):
    # sid for identifying the client connected
    print("Client connected")


@sio_server.event
def disconnect(sid):
    # implement this function, if disconnected
    print("Client disconnected")

def periodic_task():
    while True:
        # 随机生成控制参数
        throttle = random.uniform(0, 1)  # 随机油门值
        brake = 0  # 随机刹车值
        steering_angle = random.uniform(-1, 1)  # 随机转向角，范围从-1到1

        # 发送控制命令
        send_control(throttle, brake, steering_angle)

        # 打印发送的控制命令，以便跟踪
        print(f"Sent control - Throttle: {throttle}, Brake: {brake}, Steering Angle: {steering_angle}")

        # 等待一定时间，这里假设是每2秒发送一次控制命令
        eventlet.sleep(2)

# Defining Listened Events and Associated Methods
@sio_server.on("send_image")
def process_image(sid, data):
    # print("image data from client recieved!"
    # here you write what you want to do with the received data. e.g. give the image to the neural network model for command prediction
    img_data = data["image"]
    img_bytes = base64.b64decode(img_data)
    # Decode image from base64 format
    img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

    if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
        cv2.namedWindow("image from unity", cv2.WINDOW_NORMAL)
        cv2.imshow("image from unity", img)
    else:
        print("Invalid image data")

    steering_angle = 0.25
    throttle = 0.25
    brake = 0

    send_control(throttle, brake, steering_angle)


@sio_server.on("vehicle_data")
def vehicle_command(sid, data):
    print("vehicle data recieved!")
    # here you write what you want to do with the received data.
    throttle = float(data["throttle"])
    brake = float(data["brake"])
    steering_angle = float(data["steering_angle"])
    velocity = float(data["velocity"])

    print(f"velocity of the car: {velocity}")


# Defining the Method to sending Control Command back to Client
def send_control(throttle, brake, steering_angle):
    control_data = {"throttle": throttle.__str__(),
                    "brake": brake.__str__(),
                    "steering_angle": steering_angle.__str__()}

    sio_server.emit("control_command", data=control_data, skip_sid=True)


if __name__ == "__main__":
    eventlet.spawn(periodic_task)
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio_server, app)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)
