# 🚶‍♂️ Pedestrian Detection System (AAI3001 - Deep Learning & Computer Vision)
**Team 11**

A complete two-term project exploring **image classification** and **object detection** for pedestrian-related safety applications.  
This repository contains an end-to-end **Streamlit** web application featuring:

- 🖼 **Term 1** - Pedestrian **Classification** (ResNet-18)  
- 🛴 **Term 2** - Pedestrian **Object Detection** (RetinaNet-ResNet50 FPN)  
- 🔥 **Grad-CAM explanations**, **occlusion analysis**, **visualisation tools**  
- 📩 Optional **Email / Telegram alert pipeline**  
- ☁️ Cloud hosting through Hugging Face + Render

---

## 🌐 Live Demos

| Feature | Link |
|--------|------|
| ☁️ **Cloud Object Detection Demo** | https://huggingface.co/spaces/y-yingg/pedestrian_detection-demo |
| 🌍 **Information Website / Full Streamlit App** | https://aai3001-t1-gc.onrender.com/ |
| 🗂 **Dataset (Roboflow)** | https://universe.roboflow.com/atech-witjl/pedestrian-kmhf3/dataset/3 |

---
## 🌐 Term 2 Object Detection Model .pth file

| Model | Link |
|--------|------|
| ☁️ **Model.pth** | https://huggingface.co/y-yingg/pedestrian_detection/tree/main | 
---

# 📁 Project Structure


```bash
📦 AAI3001/
├─ 📂 data-object_detection/
│  ├─ 📂 train/
│  └─ 📂 validation/
├─ 📂 data/
│  ├─ 📂 train/
│  │  ├─ 📂 no pedestrian/
│  │  ├─ 📂 pedestrian/
│  │  └─ labels.csv
│  └─ 📂 validation/
│     ├─ 📂 no pedestrian/
│     ├─ 📂 pedestrian/
│     └─ labels.csv
├─ 📂 models/
│  ├─ best_resnet18_pedestrian.pt      # term 1 classification model
│  └─ resnet18_pedestrian.pt      # term 1 classification model
├─ 📂 pages/
│  ├─ 0_Object Detection - Project Detail.py
│  ├─ 1_Object Detection - Model.py
│  ├─ 2_Object Detection - Model Explanations.py
│  ├─ 3_Classification - Project Detail.py
│  ├─ 4_Classification - (Best Model).py
│  ├─ 5_Classfication - (Base Model).py
│  ├─ 6_ Classification - Models Explanations.py
│  ├─ 7_Classfication - Saliency.py
│  └─ 8_Telegram Help.py
├─ .dockerignore
├─ .gitignore
├─ Dockerfile
├─ Home.py
├─ Project1.ipynb
├─ Project1DataCollection.ipynb
├─ README.md
├─ requirements.txt
├─ test.py
├─ requirements.txt
├─ tuned_retinanet_training.ipynb
└─ utils.py
````


Key modules include:  
- **Classification (Base ResNet-18)** -`5_Classification - (Base Model).py` :contentReference[oaicite:0]{index=0}  
- **Classification (Best ResNet-18 + TTA + threshold tuning)** -`4_Classification - (Best Model).py` :contentReference[oaicite:1]{index=1}  
- **Object Detection (RetinaNet ResNet-50 FPN)** -`2_Object Detection Model.py` :contentReference[oaicite:2]{index=2}  
- **Saliency maps (gradients + occlusion)** -`7_Classification - Saliency.py` :contentReference[oaicite:3]{index=3}  
- **Model explanation pages** -`6_Classification - Models Explanations.py`, `Object Detection - Explanation` :contentReference[oaicite:4]{index=4}

---

# 🧠 Project Overview

This two-term project explores how deep learning can detect pedestrians for safety, monitoring, or alert systems.

## **Term 1 - Pedestrian Classification**
A fine-tuned **ResNet-18** model determines whether an uploaded image contains a **pedestrian** or **no pedestrian**.

Features:
- Transfer learning on ResNet-18  
- Two models:
  - **Base model** (ImageNet pretrained)  
  - **Best model** (balanced sampling, label smoothing, RandomErasing, two-phase training)
- TTA (Test-Time Augmentation)
- Adjustable decision threshold slider
- Integrated **Telegram** and **Email alerts**
- Gradient saliency maps + Occlusion sensitivity maps

## **Term 2 - Object Detection**
A full extension to object detection using **RetinaNet (ResNet-50 FPN)**:

Detects:
- 🚴 Person riding bicycle  
- 🛴 Person riding kickboard  
- 🏍 Person riding motorcycle  

Features:
- COCO-style dataset  
- RetinaNet + FPN  
- Full detection pipeline with bounding boxes and labels  
- Grad-CAM applied directly on detector outputs  
- Example browser reading images from validation folder  
- HuggingFace-hosted model for cloud deployment

---

# 🚀 Quick Start

## 1️⃣ Clone the repository
```bash
git clone https://github.com/y-yingg/AAI3001.T1.GC.git
````

## 2️⃣ Install dependencies

(from `requirements.txt`) 

```bash
pip install -r requirements.txt
```

## 3️⃣ Run the Streamlit application

```bash
streamlit run Home.py
```

All pages will be visible on the left sidebar.

---

# 🧰 Technologies Used

### **Deep Learning**

* PyTorch
* ResNet-18 (classification)
* RetinaNet (ResNet-50 FPN)
* Grad-CAM / Occlusion sensitivity

### **Web Deployment**

* Streamlit
* Hugging Face Hub
* Render.com
* Telegram Bot API
* Gmail SMTP

### **Others**

* NumPy, Pandas
* PIL
* Matplotlib
* joblib
* torchvision transforms
* Custom dataset loaders (CSV-based)

---

# 🖥 Features Overview

## ⭐ **1. Pedestrian Classification (Term 1)**

### Base Model (ResNet-18)

Source: `5_Classification - (Base Model).py` 

* Simple augmentation (flip, rotation)
* Learning rate: 1e-4
* AdamW optimizer
* CrossEntropyLoss
* Email + Telegram alerts

### Best Model (Fine-tuned ResNet-18)

Source: `4_Classification - (Best Model).py` 
Enhancements:

* Two-phase training (freeze → unfreeze)
* WeightedRandomSampler
* Label smoothing
* RandomErasing
* TTA (flip)
* Threshold slider

### Visual Explainability

From `7_Classification - Saliency.py` 

* Saliency maps (gradient-based)
* Occlusion sensitivity
* Heatmap overlays
* Critical region marking

---

## ⭐ **2. Object Detection (Term 2)**

### Model

Source: `2_Object Detection Model.py` 

* RetinaNet ResNet-50 FPN
* Trained on 512×512 images
* Custom transforms & normalization
* Loaded from HuggingFace model repository

### Outputs

* Bounding boxes
* Confidence scores
* Detection table
* Grad-CAM directly on detection score tensors
* Sample validation images preview

---

## ⭐ **3. Additional Pages**

### Term 1 vs Term 2 Comparison

A high-level UI summary showing the progression of the project.
Source: `pages/Home_T1_T2_Comparison.py`  *(example)*

### Model Explanation Pages

* Classification model explanation page
* Object detection explanation page
  (architectures, training methods, metrics)

---

# 📊 Performance Summary

### **Best ResNet-18 Classification Model**

* High validation accuracy
* Controlled false negatives via weighted loss
* Interpretable via saliency & occlusion

### **RetinaNet Object Detection Performance**

From object detection explanation page:

* **AP@[0.5:0.95] = 0.678**
* **AP50 ≈ 0.915**
* **AP75 ≈ 0.799**
* Good medium/large object performance
* Lower small-object AP due to dataset and image scale

---

# 🏗 Deployment Guide

### Deploy on Hugging Face

* Upload model `.pth` to HuggingFace Hub
* Use `hf_hub_download()` in Streamlit to load it
  (implemented in detection page)

### Deploy on Render.com

* Create a web service
* Use `streamlit run Home.py` as startup command
* Ensure `requirements.txt` is installed

---

# 🔒 Security Notes

* API keys (Telegram/SMTP) are entered via UI → **not stored in code**
* Avoid committing `.env` or tokens to Git
* Email sending uses Gmail App Passwords (secure method)

---

# 📌 Future Improvements

* Multi-class pedestrian behaviour understanding
* Real-time CCTV video inference
* YOLOv8/YOLOv11 comparison
* Edge device optimisation (Jetson Nano / Coral TPU)
* Better small-object recall with richer datasets
* Tracking (DeepSORT / StrongSORT)

---


