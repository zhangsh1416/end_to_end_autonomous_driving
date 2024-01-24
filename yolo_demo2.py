import torch
from PIL import Image, ImageDraw
import os

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)

# Path to your image (you'll replace this with the path to your image)
image_path = '/home/shihong/桌面/IMG_1452.JPG'  # Replace with your image path

# Load image
img = Image.open(image_path)

# Perform inference
results = model(img)

# Process results
for *box, conf, cls in results.xyxy[0].cpu().numpy():
    # Draw box
    ImageDraw.Draw(img).rectangle(box, outline='red', width=3)

# Determine the format of the input image
_, ext = os.path.splitext(image_path)
output_format = ext[1:].upper()  # Convert file extension to upper case for PIL compatibility

# Correct format for JPEG
if output_format == 'JPG':
    output_format = 'JPEG'

# Save or display the image
output_path = image_path.replace(ext, '_detected' + ext)
img.save(output_path, format=output_format)

print(f"Detection completed. Image saved at {output_path}")
