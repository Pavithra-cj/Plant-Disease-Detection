import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from PIL import Image

# -------------------------------
# Load model & class names
# -------------------------------
MODEL_PATH = "plant_disease_model.h5"
CLASS_JSON = "class_names.json"

model = load_model(MODEL_PATH)

with open(CLASS_JSON, 'r') as f:
    class_names = json.load(f)

# -------------------------------
# Prediction function
# -------------------------------
def predict_leaf(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    preds = model.predict(x)
    class_idx = np.argmax(preds[0])
    confidence = preds[0][class_idx]
    return class_names[class_idx], confidence

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="🌿 Plant Disease Detector", page_icon="🍃", layout="wide")

st.markdown(
    """
    <div style="text-align:center">
        <h1 style="color:#27AE60;">🌿 Plant Disease Detector</h1>
        <h3 style="color:#145A32;">AI-Powered Leaf Image Classification</h3>
        <hr style="border:1px solid #27AE60">
    </div>
    """, unsafe_allow_html=True
)

col1, col2 = st.columns([1,2])

with col1:
    uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg","png","jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

with col2:
    if uploaded_file:
        # Save temp file
        with open("temp.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Predict
        pred_class, confidence = predict_leaf("temp.jpg")

        # Display result
        st.markdown(f"### 🧠 Predicted Disease: *{pred_class}*")
        st.markdown(f"### 📊 Confidence: *{confidence*100:.2f}%*")
        st.progress(int(confidence * 100))

st.markdown(
    """
    <hr style="border:1px solid #27AE60">
    <p style="text-align:center; color:gray;">Powered by Deep Learning 🌱</p>
    """, unsafe_allow_html=True
)
