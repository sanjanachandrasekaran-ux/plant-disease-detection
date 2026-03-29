import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json
import os
import gdown

# -------------------------------
# Download model
# -------------------------------
file_id = "1Deg-sxG1v2Ezi48C5RkRFamyM-KnNYzl"
model_path = "plant_disease_model.h5"

if not os.path.exists(model_path):
    with st.spinner("Downloading model... ⏳"):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)

# -------------------------------
# Load model
# -------------------------------
model = load_model(model_path)

# -------------------------------
# Load JSON
# -------------------------------
with open('treatments.json', 'r') as f:
    treatments = json.load(f)

class_names = list(treatments.keys())

# -------------------------------
# UI Title
# -------------------------------
st.markdown("<h1 style='text-align: center; color: green;'>🌱 Plant Disease Detection</h1>", unsafe_allow_html=True)

st.write("Upload or capture a plant leaf image to detect disease.")

# -------------------------------
# 🌟 TOGGLES
# -------------------------------
use_camera = st.toggle("📸 Use Camera")
use_tamil = st.toggle("🌐 Enable Tamil Language")
show_confidence = st.toggle("🔍 Show Confidence")

# -------------------------------
# Image Input Logic
# -------------------------------
image = None

if use_camera:
    camera_image = st.camera_input("Take a photo")
    if camera_image is not None:
        image = camera_image
else:
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = uploaded_file

# -------------------------------
# Prediction
# -------------------------------
if image is not None:
    img = Image.open(image)
    st.image(img, caption="Input Image", use_column_width=True)

    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing... 🌿"):
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        disease = class_names[class_index]
        confidence = np.max(prediction) * 100

    clean_name = disease.replace("_", " ")

    # Tamil label
    if use_tamil:
        clean_name = "நோய்: " + clean_name

    st.subheader(f"🦠 {clean_name}")

    # Confidence toggle
    if show_confidence:
        st.info(f"🔍 Confidence: {confidence:.2f}%")

    if confidence < 70:
        st.warning("⚠️ Low confidence. Try another image.")

    # Healthy / Diseased
    if "healthy" in disease.lower():
        st.success("✅ Plant is Healthy")
    else:
        st.error("❌ Plant is Diseased")

    # Language selection
    lang_code = "ta" if use_tamil else "en"

    st.success(f"💊 Treatment: {treatments[disease]['treatment'][lang_code]}")
    st.info(f"🌿 Fertilizer: {treatments[disease]['fertilizer'][lang_code]}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("🌿 Built with Streamlit | AI Project by Sanju")