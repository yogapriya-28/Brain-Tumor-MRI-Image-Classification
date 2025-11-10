# 🧠 Brain Tumor MRI Classification using Deep Learning

A deep learning–based project that classifies **Brain MRI images** into four tumor categories using **InceptionV3**, **MobileNetV2**, **ResNet50**, and a **Custom CNN** model.  
The final model (`InceptionV3_best.h5`) is deployed as an interactive **Streamlit web app** for real-time tumor diagnosis.

---

## 📘 Project Overview

This project applies **Convolutional Neural Networks (CNNs)** and **Transfer Learning** to detect and classify brain tumors from MRI scans.  
The task involves preprocessing MRI images, training multiple architectures, evaluating their performance, and deploying the best-performing model in a Streamlit interface.

---

## 🧩 Dataset

**Source:** Brain Tumor MRI Dataset 
**Classes:**
- 🧬 Glioma  
- 🧬 Meningioma  
- 🧬 No Tumor  
- 🧬 Pituitary Tumor  

**Dataset Split:**
- Training Set: 1695 images  
- Validation Set: 502 images  
- Test Set: 246 images  

All images were resized to **224 × 224 px** and normalized to `[0,1]`.

---

## 🧠 Models Trained

| Model | Accuracy | Precision | Recall | F1-Score | Remarks |
|--------|-----------|------------|----------|-----------|----------|
| Custom CNN | 0.2479 | 0.1451 | 0.2480 | 0.1319 | Underfitting |
| ResNet50 | 0.4431 | 0.5702 | 0.4431 | 0.3257 | Moderate performance |
| MobileNetV2 | 0.6829 | 0.7328 | 0.6829 | 0.6722 | Lightweight and efficient |
| **InceptionV3** | **0.7033** | **0.7194** | **0.7033** | **0.6967** | 🏆 Best performing |

---

## 🏆 Final Model – InceptionV3

- Accuracy: **70.3%**
- Model file: `models_outputs/InceptionV3_best.h5`
- Framework: **TensorFlow / Keras**
- Optimizer: Adam 
- Loss: Categorical Crossentropy
- Epochs: 15  
- Image Size: (224, 224, 3)

---

## 📊 Class-wise Performance (InceptionV3)

| Class | Precision | Recall | F1-Score |
|--------|------------|----------|----------|
| Glioma | 0.75 | 0.81 | 0.78 |
| Meningioma | 0.63 | 0.46 | 0.53 |
| No Tumor | 0.91 | 0.63 | 0.75 |
| Pituitary | 0.61 | 0.89 | 0.72 |

---

## 🚀 Streamlit Deployment

### 📂 File: `brain_tumor_app.py`

A real-time, lightweight **Streamlit app** for MRI classification.  
Users can upload an MRI image (`.jpg`, `.jpeg`, `.png`), and the model predicts the tumor type with confidence scores.

### 🧭 Run Locally

```bash
# 1. Clone repository
git clone https://github.com/your-username/brain-tumor-classification.git
cd brain-tumor-classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run brain_tumor_app.py
