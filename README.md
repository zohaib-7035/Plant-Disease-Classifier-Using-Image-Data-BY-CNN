

# 🌱 Plant Disease Classifier - Streamlit App

This is a web-based deep learning app built using **Streamlit** that allows users to detect plant diseases from leaf images. It leverages a trained **Convolutional Neural Network (CNN)** model on the PlantVillage dataset to identify 38 different plant conditions (including healthy and diseased states).

---
---

## Trained Model Link is given below
https://drive.google.com/file/d/1rKh-IElSdHTqax7XdfSdZTn-r8T_qWPf/view?usp=drive_link)https://drive.google.com/file/d/1rKh-IElSdHTqax7XdfSdZTn-r8T_qWPf/view?usp=drive_link

---
---
## 🔍 Features

* 🌿 Detect plant diseases from uploaded leaf images.
* 🧪 Shows **prediction confidence**.
* 📊 Displays **class probabilities** as a horizontal bar chart.
* 💡 Clean dark theme with Orbitron animated UI.
* 📷 Image is resized and previewed before prediction.

---

## 📁 Project Structure

```
.
├── app.py                         # Streamlit web app
├── plant_disease_prediction_model.h5  # Trained Keras CNN model
├── class_indices.json             # Label dictionary (class index to class name)
├── requirements.txt               # List of required Python packages
├── README.md                      # This file
```

---

## 🔧 Installation

### 🔹 Clone the repository:

```bash
git clone https://github.com/yourusername/plant-disease-classifier.git
cd plant-disease-classifier
```

### 🔹 Install the dependencies:

```bash
pip install -r requirements.txt
```

> Make sure Python 3.8+ is installed.

---

## 🚀 Running the App

```bash
streamlit run app.py
```

* Upload a **leaf image** (JPG or PNG).
* Click **🔍 Predict**.
* View the predicted class, confidence, and probability chart.

---

## 🧠 Model Details

* ✅ Trained on: [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
* 🔢 Classes: 38 (healthy + diseased conditions of crops)
* 🏗️ Architecture: CNN with 3 Conv2D layers + MaxPooling + Dense layers
* 🎯 Accuracy: \~95% on validation data

---

## 📚 Example Classes

* Apple\_\_\_Black\_rot
* Grape\_\_\_Esca\_(Black\_Measles)
* Tomato\_\_\_Late\_blight
* Potato\_\_\_healthy
* Pepper,\_bell\_\_\_Bacterial\_spot
* ...and 30+ more

---


---

## 📜 License

MIT License — use freely with attribution.

---

