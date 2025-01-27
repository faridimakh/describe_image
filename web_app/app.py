import os

import torch
from PIL import Image
from flask import Flask, request, render_template, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration, MarianMTModel, MarianTokenizer

app = Flask(__name__)

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load MarianMT translation model
translation_model_name = "Helsinki-NLP/opus-mt-en-fr"
translator = MarianMTModel.from_pretrained(translation_model_name)
tokenizer = MarianTokenizer.from_pretrained(translation_model_name)


# Route for the main page
@app.route('/')
def index():
    return render_template('home.html')


# Route for handling image upload and caption generation
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save and process the uploaded image
    image_path = os.path.join("../uploads", file.filename)
    file.save(image_path)
    image = Image.open(image_path).convert("RGB")

    # Generate caption using BLIP model
    inputs = processor(image, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, min_length=10, max_length=400)
    description_en = processor.decode(outputs[0], skip_special_tokens=True)

    # Translate caption to French
    translation_inputs = tokenizer(description_en, return_tensors="pt", truncation=True)
    translation_outputs = translator.generate(**translation_inputs)
    description_fr = tokenizer.decode(translation_outputs[0], skip_special_tokens=True)

    # Clean up the uploaded file
    os.remove(image_path)

    return jsonify({"description_en": description_en, "description_fr": description_fr})


if __name__ == '__main__':
    os.makedirs("../uploads", exist_ok=True)
    app.run(host='0.0.0.0')
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# pip install sentencepiece
# pip install pillow
