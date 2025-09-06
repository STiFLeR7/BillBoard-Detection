# 🏙️ Billboard Detection using YOLOv8

An **AI-powered computer vision system** to detect and flag billboards from images/videos, enabling scalable enforcement of outdoor advertising policies. This repo contains the **training pipeline, inference utilities, and deployable model artifacts** (ONNX, PyTorch, TFLite-ready) for seamless integration into mobile or backend applications.

---

## 🚀 Features

* **YOLOv8-based Detector** trained on Roboflow's *Billboards Detection* dataset.
* **mAP50 ≈ 0.89** after 50 epochs on RTX 3050 6GB GPU.
* Exports to **ONNX, PyTorch, TFLite-ready** for cross-platform deployment.
* Modular repo structure: training, inference, exports, Docker support.
* Compatible with **mobile apps** (Android/iOS) or **server APIs** (FastAPI, Docker).

---

## 📂 Repository Structure

```
BillBoard-Detection/
│── data/                      # Dataset (train/val/test + data.yaml)
│── models/                    # Saved model checkpoints (.pt, .onnx, .tflite)
│── exports/                   # Future-ready deployment exports
│── src/
│   ├── datasets/              # Data preparation utilities
│   ├── models/                # Model definitions
│   ├── training/              # Training scripts
│   │   └── train.py           # Training automation script
│   ├── inference/             # Inference scripts & API
│   │   ├── infer.py           # Run inference on images
│   │   └── api.py             # FastAPI server for Docker deployment
│   └── utils/                 # Helper functions
│── docker/                    # Dockerfiles for deployment
│── runs/                      # YOLOv8 training logs & results
│── tests/                     # Unit tests
│── README.md                  # Project documentation
│── requirements.txt            # Dependencies
```

---

## 📊 Training Results

* **Precision (P):** 0.886
* **Recall (R):** 0.825
* **mAP\@50:** 0.889
* **mAP\@50-95:** 0.768
* **Model Size:** \~6.2 MB (`best.pt`)
* **Speed:** \~3.3ms inference per image (RTX 3050)

---

## ⚡ Quickstart

### 🔹 Train Model

```powershell
python src/training/train.py
```

* Saves best model → `models/best.pt`
* Training logs → `runs/detect/train/`

### 🔹 Run Inference (Images)

```powershell
python src/inference/infer.py
```

* Input: images from `val/images/`
* Output: annotated results in `runs/inference/`

### 🔹 Export for Deployment

```powershell
# ONNX
yolo export model=models/best.pt format=onnx

# TFLite (Linux/Docker recommended)
yolo export model=models/best.pt format=tflite
```

---

## 🐳 Docker Deployment (ONNX Runtime + FastAPI)

### Build Image

```bash
docker build -t billboard-api -f docker/Dockerfile .
```

### Run Container

```bash
docker run -p 8000:8000 billboard-api
```

Access API at 👉 `http://localhost:8000/docs`

---

## 📱 App Integration Guide

* **Input:** RGB image, resized to 640×640.
* **Output:** Bounding boxes `[x1, y1, x2, y2, confidence, class_id]`.
* **Classes:**

  * `0 → billboard`
* **Preferred Format for Mobile:** `best.onnx`
* **Alternative Format (Android):** `best.tflite`

ONNX Runtime is available for **Android, iOS, C++, Python** → recommended for production.

---

## 🛠️ Tech Stack

* **Framework:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* **Language:** Python 3.11
* **DL Backend:** PyTorch 2.7.1 + CUDA 12.6
* **Deployment:** ONNX Runtime, TensorFlow Lite, FastAPI, Docker
* **Hardware Used:** NVIDIA RTX 3050 Laptop GPU (6GB)

---

## 📌 Future Enhancements

* 📍 Geotagging support (GPS metadata + violation mapping)
* 📏 Billboard dimension validation (3D estimation)
* 🔍 Content moderation (OCR + obscenity/misinformation filter)
* 🌐 Public dashboard with heatmaps of flagged billboards

---

## 👨‍💻 Authors

* **ML Pipeline:** [Stifler](https://github.com/STiFLeR7)
* **App Development:** *(to be integrated by teammate)*

---

## 📜 License

This project is for **educational & research purposes** (Smart City AI Challenge). Ensure ethical AI usage & compliance with data privacy laws.
