import os
from flask import Flask, request, render_template, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
import requests

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained ResNet model from torchvision
model = models.resnet50(pretrained=True)
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Home route to render image upload page
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route for image classification
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    img = Image.open(file.stream)

    # Apply image transformation
    img_tensor = transform(img).unsqueeze(0)

    # Get prediction from model
    with torch.no_grad():
        output = model(img_tensor)

    _, predicted_class = torch.max(output, 1)
    predicted_class = predicted_class.item()

    # Get class labels from ImageNet
    LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    labels = requests.get(LABELS_URL).json()
    class_label = labels[str(predicted_class)][1]

    # Return the result to the frontend
    return render_template('index.html', image_path=file.filename, prediction=class_label)

if __name__ == '__main__':
    app.run(debug=True)
