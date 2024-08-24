import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd

# Load the pre-trained models
model_gender = tf.keras.models.load_model('best_models/gender_model.keras')
model_race = tf.keras.models.load_model('best_models/race_model.keras')
model_age = tf.keras.models.load_model('best_models/age_model.keras')

# Preprocessing function
def preprocess_image(image, target_size=(128, 128)):
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Prediction functions
def predict_gender(image):
    prediction = model_gender.predict(image).ravel()[0]
    return "Male" if prediction < 0.5 else "Female"

def predict_race(image):
    prediction = model_race.predict(image)
    races = ["Black", "Indian", "Others", "White"]
    return races[np.argmax(prediction)]

def predict_age(image):
    prediction = model_age.predict(image).ravel()[0]
    return round(prediction)

# Streamlit app interface
st.title("Demographic Feature Detection")
st.write("Upload an image to get predictions for gender, race, and age.")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make predictions
    gender = predict_gender(processed_image)
    race = predict_race(processed_image)
    age = predict_age(processed_image)

    # Display predictions
    st.write(f"**Predicted Gender:** {gender}")
    st.write(f"**Predicted Race:** {race}")
    st.write(f"**Predicted Age:** {age}")
