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
st.markdown("<h1 style='text-align: center; color: green;'>🌱 Plant Disease Detection</h1>", unsafe_allow_html=True)

st.write("Upload a plant leaf image to detect disease and get treatment suggestions.")

# Upload + Camera
uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "png", "jpeg"])
camera_image = st.camera_input("📸 Or take a photo")

# Decide which image to use
image = None
if uploaded_file is not None:
    image = uploaded_file
elif camera_image is not None:
    image = camera_image

# -------------------------------
# Prediction
# -------------------------------
if image is not None:
    img = Image.open(image)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing leaf... 🌿"):
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        disease = class_names[class_index]
        confidence = np.max(prediction) * 100

    clean_name = disease.replace("_", " ")

    st.subheader(f"🦠 Predicted Disease: {clean_name}")
    st.info(f"🔍 Confidence: {confidence:.2f}%")

    if confidence < 70:
        st.warning("⚠️ Prediction confidence is low. Try another image.")

    if "healthy" in disease.lower():
        st.success("✅ Plant is Healthy")
    else:
        st.error("❌ Plant is Diseased")

    st.success(f"💊 Treatment: {treatments[disease]['treatment']}")
    st.info(f"🌿 Fertilizer: {treatments[disease]['fertilizer']}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("🌿 Built with Streamlit | AI Plant Disease Detection")