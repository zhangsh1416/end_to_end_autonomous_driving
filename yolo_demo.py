import torch
from PIL import Image, ImageDraw
import os

# Function to load and process the image
def load_image(image_path):
    image = Image.open(image_path)
    return image

# Function to perform object detection using YOLOv5s
def detect_objects(image_path, model):
    # Load and preprocess the image
    image = load_image(image_path)
    results = model(image)
    return results

# Function to draw bounding boxes on the image
def draw_boxes(results, image_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    for *xyxy, conf, cls in results.xyxy[0]:
        draw.rectangle(xyxy, outline="red", width=3)
    return image

# Placeholder for the image path
image_path = '/home/shihong/桌面/DSC07655.JPG'  # Replace with actual image path

# Load YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Perform object detection
results = detect_objects(image_path, model)

# Draw bounding boxes on the image
image_with_boxes = draw_boxes(results, image_path)

# Save the image with bounding boxes
output_path = os.path.join(os.path.dirname(image_path), 'detected_' + os.path.basename(image_path))
image_with_boxes.save(output_path)

# Output path of the saved image
print("Saved detected image at:", output_path)
