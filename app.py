import os
from pathlib import Path
from flask import Flask, redirect, render_template, request, jsonify
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model

# Load disease and supplement information
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Load the trained TensorFlow model
model = load_model("model_epoch_16.keras")  # Update with your trained model path

# Define the class labels (adjust as per your model's class mapping)
idx_to_classes = {0: 'Early_blight', 1: 'Healthy', 2: 'Late_blight', 3: 'Leaf Miner', 4: 'Magnesium Deficiency', 5: 'Nitrogen Deficiency', 6: 'Spotted Wilt Virus'}

# Prediction function
def prediction(image_path):
    img = keras_image.load_img(image_path, target_size=(128, 128))  # Resize image
    img_array = keras_image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize image (important if the model was trained with normalization)
    
    predictions = model.predict(img_array)  # Predict the class
    index = np.argmax(predictions)  # Get the predicted class index
    return index

# Weather API setup (using OpenWeatherMap API)
import requests

API_KEY = "c80d63cb1ed3f119d3befcf349e6ebb8"  # Replace with your real key
CITY = "Coimbatore"

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    
    print(data)

    if response.status_code != 200 or "main" not in data:
        return {
            "error": f"Weather fetch failed: {data.get('message', 'Unknown error')}"
        }

    temp = data['main']['temp']
    humidity = data['main']['humidity']
    condition = data['weather'][0]['description'].title()

    if humidity > 80 and temp < 22:
        alert = "🚨 High risk of Late Blight. Take precautions!"
    elif temp > 30 and humidity < 40:
        alert = "⚠️ Dry weather — check for Leaf Miner."
    else:
        alert = "✅ Weather conditions normal. No major risk."

    return {
        "temp": temp,
        "humidity": humidity,
        "condition": condition,
        "alert": alert,
        "city": city
    }

# Flask app setup
app = Flask(__name__)

@app.route('/')
def home_page():
    weather = get_weather(CITY)
    return render_template('home.html', weather=weather)

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        if not filename or '.' not in filename:
            return "Invalid image", 400

        # Save image securely
        upload_dir = Path("static/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / filename
        image.save(file_path)

        # Get the prediction
        pred = prediction(str(file_path))
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]

        return render_template('submit.html', title=title, desc=description, prevent=prevent,
                               image_url=image_url, pred=pred, sname=supplement_name,
                               simage=supplement_image_url, buy_link=supplement_buy_link)

@app.route('/genie')
def tomato_genie_page():
    return render_template('Genie.html')

@app.route('/chat', methods=['POST'])
def chat_response():
    data = request.get_json()
    user_prompt = data.get("prompt", "")
    if not user_prompt:
        return jsonify({"reply": "Please ask a question about tomato plant disease."})
    raw_reply = chatquery(user_prompt)
    formatted_reply = raw_reply.replace('\n', '<br>')  # Convert line breaks to <br>
    return jsonify({'reply': formatted_reply})

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']), disease=list(disease_info['disease_name']), buy=list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
