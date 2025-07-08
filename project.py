# app.py

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import json

# Page setup
st.set_page_config(page_title="🌿 Plant Disease Classifier", layout="centered")

# Load model
model = load_model('plant_disease_prediction_model.h5')

# Load class names
with open("class_indices.json", "r") as f:
    class_names = json.load(f)
    class_names = {int(k): v for k, v in class_names.items()}

# Orbitron dark theme styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&display=swap');

html, body, .stApp {
    background: linear-gradient(145deg, #2e2e2e, #1e1e1e);
    color: white;
    font-family: 'Orbitron', sans-serif;
    overflow-x: hidden;
}

h1 {
    text-align: center;
    font-size: 3rem;
    background: linear-gradient(to right, #a8ff78, #78ffd6, #a8ff78);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 3s infinite;
}

@keyframes shine {
    0% {background-position: 0%;}
    100% {background-position: 200%;}
}

.instruction {
    text-align: center;
    font-size: 1.2rem;
    color: #ccc;
    margin-bottom: 20px;
    animation: fadeIn 2s ease-in-out;
}

.uploaded-img {
    display: flex;
    justify-content: center;
    margin-top: 20px;
    animation: fadeInZoom 1s ease-out;
}

img:hover {
    box-shadow: 0 0 20px #9efff3;
    border-radius: 10px;
    transform: rotate(1deg) scale(1.1);
    transition: 0.5s ease-in-out;
}

.prediction-box {
    margin-top: 30px;
    padding: 25px;
    border-radius: 15px;
    background: rgba(255, 255, 255, 0.05);
    box-shadow: 0 8px 32px 0 rgba( 31, 38, 135, 0.37 );
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.18);
    text-align: center;
    font-size: 1.5rem;
    color: #00ffe7;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% {box-shadow: 0 0 10px #00ffe7;}
    100% {box-shadow: 0 0 25px #00ffe7;}
}

@keyframes fadeInZoom {
    0% {transform: scale(0.8); opacity: 0;}
    100% {transform: scale(1); opacity: 1;}
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>🌱 Plant Disease Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='instruction'>Upload a leaf image (224x224) to detect the plant disease</p>", unsafe_allow_html=True)

# Upload Image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((224, 224))
    st.markdown("<div class='uploaded-img'>", unsafe_allow_html=True)
    st.image(image, caption="Uploaded Image (Resized)", width=250)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔍 Predict"):
        # Preprocess
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_label = class_names[predicted_index]
        confidence = np.max(predictions[0]) * 100

        # Result box
        st.markdown(f"<div class='prediction-box'>🌿 <b>{predicted_label}</b><br>🧪 Confidence: <b>{confidence:.2f}%</b></div>", unsafe_allow_html=True)

        # Confidence bar chart
        st.markdown("### 🌡️ Class Probabilities")
        fig, ax = plt.subplots(figsize=(6, 8))
        probs = tf.nn.softmax(predictions[0]).numpy()
        labels = [class_names[i] for i in range(len(probs))]
        ax.barh(labels, probs, color='#00ffe7')
        ax.set_xlim([0, 1])
        ax.set_xlabel('Confidence')
        ax.invert_yaxis()
        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#2e2e2e')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        st.pyplot(fig)
