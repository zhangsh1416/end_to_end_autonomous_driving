# Server-to-Client Client
import socketio
import numpy as np

sioc = socketio.Client()

@sioc.event
def connect():
    print('connected to server')

@sioc.event
def data2control(data):
    image = np.random.randint(256, size=(480, 640, 3)).astype('uint8')
    sioc.emit(event = 'ai_driver', data=image.tobytes())
    print('received')

# Send image from client to server.
@sioc.event
def control(data):
    print('controled')
        
@sioc.event
def disconnect():
    quit()

if __name__ == '__main__':
    sioc.connect('http://localhost:5000')
    sioc.wait()