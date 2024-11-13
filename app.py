import os
from flask import Flask, request, render_template, send_from_directory
import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import requests

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the pre-trained ResNet model
model = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and classification
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return {"error": "No image file provided"}, 400

    file = request.files['image']
    filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    img = Image.open(file_path)
    img_tensor = transform(img).unsqueeze(0)

    # Perform prediction
    with torch.no_grad():
        output = model(img_tensor)
    _, predicted_class = torch.max(output, 1)
    predicted_class = predicted_class.item()

    # Retrieve class label
    LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    labels = requests.get(LABELS_URL).json()
    class_label = labels[str(predicted_class)][1]

    # Annotate the image
    annotated_filename = f"annotated_{filename}"
    annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
    annotated_img = img.convert("RGB")
    draw = ImageDraw.Draw(annotated_img)

    # Optional: Load a font (adjust path and size as needed)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    text = f"Prediction: {class_label}"
    text_width, text_height = draw.textsize(text, font=font)
    position = (10, 10)  # Top-left corner for text placement with padding
    draw.rectangle(
        [position, (position[0] + text_width + 10, position[1] + text_height + 5)],
        fill="black"
    )
    draw.text((position[0] + 5, position[1]), text, fill="white", font=font)

    # Save the annotated image
    annotated_img.save(annotated_path)

    # Render the HTML template with both image paths
    return render_template(
        'index.html',
        original_image=f'static/uploads/{filename}',
        annotated_image=f'static/uploads/{annotated_filename}',
        prediction=class_label
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
