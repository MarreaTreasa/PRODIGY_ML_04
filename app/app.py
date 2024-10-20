from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model
model = load_model('simple_model.keras')

# Label and calorie maps
label_map = {
    'apple_pie': 0, 'cup_cakes': 1, 'fried_rice': 2, 'ice_cream': 3, 'macarons': 4,
    'omelette': 5, 'pizza': 6, 'ramen': 7, 'spring_rolls': 8, 'waffles': 9
}

calorie_map = {
    'apple_pie': 237,
    'cup_cakes': 292,
    'fried_rice': 238,
    'ice_cream': 137,
    'macarons': 250,
    'omelette': 154,
    'pizza': 266,
    'ramen': 188,
    'spring_rolls': 220,
    'waffles': 291
}

# Function to preprocess the image
def preprocess_image(image_data, target_size=(128, 128)):
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)  # Read the image from memory
    img = cv2.resize(img, target_size)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize the image
    return img

# Reverse the label map
reverse_label_map = {v: k for k, v in label_map.items()}

# Prediction function
def predict_food_and_calories(image_data):
    preprocessed_image = preprocess_image(image_data)

    # Predict class probabilities (classification output)
    class_probs, _ = model.predict(preprocessed_image)

    # Find the predicted class index
    predicted_class = np.argmax(class_probs, axis=1)[0]  # Single image

    # Get the food label
    predicted_label = reverse_label_map[predicted_class]

    # Get the calorie value
    predicted_calories = calorie_map[predicted_label]

    return predicted_label, predicted_calories

@app.route('/')
def upload_file():
    return render_template('index.html')  # HTML file to upload images

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    # Read the file directly into memory
    image_data = file.read()

    # Make prediction
    predicted_label, predicted_calories = predict_food_and_calories(image_data)

    return jsonify({
        'predicted_food': predicted_label,
        'predicted_calories': predicted_calories
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
