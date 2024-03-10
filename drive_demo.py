# Import the necessary libraries
import socketio
import eventlet
import time
import eventlet.wsgi
from flask import Flask
import base64
import cv2
import numpy as np
from data_preparation.datapre_for_simulator import preprocess_image_for_model
import torch
from networks.model_demo import create_custom_resnet_model
from outputs_to_labels import convert_model_output_to_labels
import threading
from threading import Event
from networks import synthetical_model
input_ready = Event()
# Create an instance of a WebSocket server based on Flask and Socket.IO.
sio_server = socketio.Server()
app = Flask(__name__)
input = None
controls = []
weights = torch.load('train/end_to_end_{timestamp}.pt')
model = create_custom_resnet_model()
model.load_state_dict(weights)
model.eval()
@sio_server.event
def connect(sid, environ):
    # sid for identifying the client connected
    # emit_empty_data()
    print("Client connected")
@sio_server.event
def disconnect(sid):
    # implement this function, if disconnected
    print("Client disconnected")
@sio_server.on("send_image")
def process_image(sid, data):
    global input
    print("image data from client recieved!")
    img_data = data["image"]
    img_bytes = base64.b64decode(img_data)
    img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    input = preprocess_image_for_model(img)
    input_ready.set()  # 标记input已准备好
    print(len(controls))
    if controls is not None:
        send_control(*controls)
@sio_server.on("vehicle_data")
def vehicle_command(sid, data):
    pass
def inference():
    global controls, input
    if not controls:
        controls = [1, 0, 0.1]
    while True:
        input_ready.wait()  # 等待input准备好
        if input is not None:
            x, y = model(input)
            x = x.squeeze(0)
            print(x)
            y = y.squeeze(0)
            controls = convert_model_output_to_labels(x, y)
            input = None  # 重置input
            input_ready.clear()  # 重置事件
        time.sleep(0.1)
        print(controls)
def send_control(throttle,  brake, steering_angle):
    control_data = [str(throttle), str(brake), str(steering_angle)]
    sio_server.emit("control_command", data=control_data, skip_sid=True)
def emit_empty_data():
    sio_server.emit('manual', data={}, skip_sid=True)
if __name__ == "__main__":

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio_server, app)
    # deploy as an eventlet WSGI server
    threading.Thread(target=inference, daemon=True).start()
    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)

