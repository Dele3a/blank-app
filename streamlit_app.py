import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


model = tf.keras.models.load_model(r'C:\Users\ghane\Project_Ghanfix\models\crack_detection_model.keras')

def preprocess_image(image, target_size=(64, 64)):
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)


st.title("Welcome, this is Ghanfix :")
st.write("Upload an image to detect if it has cracks")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.write("Classifying...")
    processed_image = preprocess_image(image, target_size=(64, 64))

    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]

    if predicted_class == 1:
        st.write("Prediction: Crack Detected")
    else:
        st.write("Prediction: No Crack Detected")



