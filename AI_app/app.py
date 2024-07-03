from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'
app.config['mnist_cnn.h5'] = 'mnist_cnn.h5'  # Path to your trained model file

# Load the trained model
model = None

def load_model_if_needed():
    global model
    if not model:
        model = load_model(app.config['mnist_cnn.h5'])

# Function to predict digits from an image file
def predict_digit_from_file(image_path):
    # Load image using Pillow (PIL)
    img = Image.open(image_path).convert('L')  # Open image in grayscale mode
    
    # Resize image to match model's expected input size (28x28)
    img = img.resize((28, 28))
    
    # Convert image to numpy array and preprocess for model input
    img_array = keras_image.img_to_array(img)
    img_array = img_array.reshape((-1, 28, 28, 1))  # Reshape for model input
    img_array /= 255.0  # Normalize pixel values
    
    # Predict digit using the loaded model
    prediction = model.predict(img_array)
    return prediction.argmax()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    load_model_if_needed()

    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    # Save the uploaded file to a temporary location
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    file.save(temp_file.name)

    # Predict digit
    predicted_digit = predict_digit_from_file(temp_file.name)

    return render_template('upload.html', file_path=temp_file.name, predicted_digit=predicted_digit)

if __name__ == '__main__':
    load_model_if_needed()
    app.run(debug=True)
