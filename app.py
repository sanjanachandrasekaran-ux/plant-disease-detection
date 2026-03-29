import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json
import os
import gdown
from gtts import gTTS
import tempfile

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Plant Disease Detector 🌱", layout="centered")

# -------------------------------
# Custom CSS (Beautiful UI)
# -------------------------------
st.markdown("""
<style>
body {
    background-color: #f5fff5;
}
h1 {
    text-align: center;
    color: #2E8B57;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
}
</style>
""", unsafe_allow_html=True)

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
# Title
# -------------------------------
st.markdown("<h1>🌱 AI Plant Disease Detection</h1>", unsafe_allow_html=True)
st.write("Upload or capture a plant leaf image to detect disease and get treatment 🌿")

# -------------------------------
# Sidebar (Settings)
# -------------------------------
st.sidebar.header("⚙️ Settings")

use_camera = st.sidebar.toggle("📸 Use Camera")
use_tamil = st.sidebar.toggle("🌐 Tamil Language")
show_confidence = st.sidebar.toggle("🔍 Show Confidence")
show_top3 = st.sidebar.toggle("🏆 Show Top 3 Predictions")
enable_voice = st.sidebar.toggle("🔊 Voice Output")

# -------------------------------
# Image Input
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
    st.image(img, caption="Input Image", use_container_width=True)

    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing... 🌿"):
        prediction = model.predict(img_array)[0]

    # Top 3
    top_indices = prediction.argsort()[-3:][::-1]
    top_results = [(class_names[i], prediction[i]*100) for i in top_indices]

    disease, confidence = top_results[0]

    clean_name = disease.replace("_", " ")
    if use_tamil:
        clean_name = "நோய்: " + clean_name

    # -------------------------------
    # Result Card
    # -------------------------------
    st.markdown("---")
    st.subheader("🧾 Result")

    st.success(f"🦠 Disease: {clean_name}")

    if show_confidence:
        st.info(f"🔍 Confidence: {confidence:.2f}%")

    if confidence < 70:
        st.warning("⚠️ Low confidence. Try another image.")

    if "healthy" in disease.lower():
        st.success("✅ Plant is Healthy")
        health_text = "Plant is healthy"
    else:
        st.error("❌ Plant is Diseased")
        health_text = "Plant is diseased"

    lang_code = "ta" if use_tamil else "en"

    treatment_text = treatments[disease]['treatment'][lang_code]
    fertilizer_text = treatments[disease]['fertilizer'][lang_code]

    st.success(f"💊 Treatment: {treatment_text}")
    st.info(f"🌿 Fertilizer: {fertilizer_text}")

    # Top 3 display
    if show_top3:
        st.subheader("🏆 Top 3 Predictions")
        for i, (d, c) in enumerate(top_results):
            name = d.replace("_", " ")
            if use_tamil:
                name = "நோய்: " + name
            st.write(f"{i+1}️⃣ {name} → {c:.2f}%")

    # Voice Output
    if enable_voice:
        lang = "ta" if use_tamil else "en"
        speech = f"{health_text}. {clean_name}. Treatment is {treatment_text}"

        tts = gTTS(text=speech, lang=lang)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            st.audio(fp.name)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("### 📘 About Project")
st.write("This AI system detects plant diseases using CNN and provides treatment suggestions.")

st.markdown("### 👩‍💻 Developed by Sanju")