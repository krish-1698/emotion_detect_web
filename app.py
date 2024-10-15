from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import os
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the trained model
model = load_model("best_model.keras")

# Mapping of class indices to class names
class_mapping = {0: 'happy', 1: 'sad', 2: 'angry', 3: 'surprised', 4: 'neutral', 5: 'disgust', 6: 'fear'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Save the uploaded image temporarily
    image_path = os.path.join('static', 'uploads', file.filename)
    file.save(image_path)

    # Preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    input_arr = np.array([img_array])

    # Make prediction
    pred = np.argmax(model.predict(input_arr))
    emotion = class_mapping[pred]

    # Display the image and the prediction result
    return render_template('results.html', emotion=emotion, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
