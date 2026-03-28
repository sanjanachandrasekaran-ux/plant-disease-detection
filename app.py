import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json
import os
import gdown

# -------------------------------
# Download model from Google Drive
# -------------------------------
file_id = "1Deg-sxG1v2Ezi48C5RkRFamyM-KnNYzl"
model_path = "plant_disease_model.h5"

if not os.path.exists(model_path):
    with st.spinner("Downloading model... please wait ⏳"):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)

# -------------------------------
# Load trained model
# -------------------------------
model = load_model(model_path)

# -------------------------------
# Load treatment data
# -------------------------------
with open('treatments.json', 'r') as f:
    treatments = json.load(f)

class_names = list(treatments.keys())

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🌱 Plant Disease Detection & Treatment System")

st.write("Upload a plant leaf image to detect disease and get treatment suggestions.")

uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    with st.spinner("Analyzing image..."):
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        disease = class_names[class_index]

    # Output
    st.subheader(f"🦠 Predicted Disease: {disease}")
    st.success(f"💊 Treatment: {treatments[disease]}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("🌿 Built with Streamlit | AI Plant Disease Detection")