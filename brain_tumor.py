# 🧠 Brain Tumor Classification Streamlit App 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time

# ------------------------------------------------------------
# 🎨 PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="🧠 Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Light elegant styling
st.markdown("""
    <style>
        /* Background */
        .stApp {
            background-color: #F5F9FF;
            color: #0D1B2A;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Title */
        h1 {
            color: #0D47A1;
            text-align: center;
            font-weight: 800;
            letter-spacing: 1px;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #E3F2FD;
            color: #0D1B2A;
            border-right: 2px solid #64B5F6;
        }

        /* Upload label */
        .stFileUploader label {
            color: #0D47A1 !important;
            font-weight: 600;
        }

        /* Buttons */
        .stButton>button {
            background-color: #1976D2;
            color: white;
            border-radius: 8px;
            font-size: 15px;
            padding: 0.5em 1em;
        }
        .stButton>button:hover {
            background-color: #0D47A1;
        }

        /* Expander */
        .stExpander {
            background-color: #E3F2FD;
            border-radius: 10px;
        }

        /* Divider */
        hr {
            border: 1px solid #64B5F6;
        }

        /* Success boxes */
        .stSuccess {
            background-color: #E3F2FD;
            border-left: 4px solid #1565C0;
        }

        /* Info */
        .stInfo {
            background-color: #E8F0FE;
            border-left: 4px solid #1E88E5;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# 🧠 MODEL CONFIG
# ------------------------------------------------------------
MODEL_PATH = r"F:\brain_tumor\models_outputs\InceptionV3_best.h5"
IMG_SIZE = (224, 224)
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# ------------------------------------------------------------
# 🧩 LOAD MODEL
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at:\n`{MODEL_PATH}`")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Model load failed: {e}")
    st.stop()

# ------------------------------------------------------------
# 🖼️ APP TITLE
# ------------------------------------------------------------
st.title("🧠 Brain Tumor Classification")
st.caption("💡 Deep Learning Model for MRI-based Tumor Diagnosis")
st.markdown("<hr>", unsafe_allow_html=True)

st.write("📤 Upload a **brain MRI image** below to classify the tumor type into one of four categories.")

# ------------------------------------------------------------
# 📤 IMAGE UPLOAD & PREDICTION
# ------------------------------------------------------------
uploaded_file = st.file_uploader("📸 Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🩺 Uploaded MRI Image", use_container_width=True)

        # Preprocess
        img_array = np.array(image)
        img_resized = tf.image.resize(img_array, IMG_SIZE)
        img_normalized = img_resized / 255.0
        img_expanded = np.expand_dims(img_normalized, axis=0)

        with st.spinner("🔍 Analyzing MRI image... Please wait..."):
            time.sleep(2)
            preds = model.predict(img_expanded)
            pred_class = np.argmax(preds, axis=1)[0]
            pred_label = CLASS_NAMES[pred_class]
            confidence = preds[0][pred_class]

        st.success("✅ Analysis Completed!")

        # Prediction results
        st.subheader("🧾 Prediction Results")
        st.markdown(f"### 🧬 **Predicted Tumor Type:** `{pred_label.capitalize()}`")
        st.write(f"**Confidence Level:** {confidence * 100:.2f}%")

        # Confidence chart
        st.markdown("### 📊 Confidence Distribution:")
        conf_data = {CLASS_NAMES[i].capitalize(): float(preds[0][i]) for i in range(len(CLASS_NAMES))}
        st.bar_chart(conf_data)

        st.info(f"🧠 The MRI image is most likely a **{pred_label.capitalize()}** tumor with {confidence*100:.2f}% confidence.")

    except Exception as e:
        st.error(f"⚠️ Error during image processing: {e}")

else:
    st.info("👆 Please upload a brain MRI image to begin classification.")

# ------------------------------------------------------------
# 📚 ABOUT SECTION
# ------------------------------------------------------------
with st.expander("ℹ️ About this App"):
    st.write("""
        This application uses a **deep learning model (InceptionV3)** trained on MRI scans 
        to classify the brain tumor type into one of four categories:
        - 🧬 Glioma  
        - 🧬 Meningioma  
        - 🧬 No Tumor  
        - 🧬 Pituitary Tumor  

        **Developed by:** *Yogapriya*  
        **Framework:** TensorFlow + Streamlit  
        **Theme:** Light Medical Blue  
        **Model File:** `InceptionV3_best.h5`  
    """)
