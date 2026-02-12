# visual_product_matcher
# 🔍 Visual Product Matcher

A FastAPI-based web app that finds visually similar products using deep learning (ResNet + FAISS).

Users can:
- Upload an image 📷  
- Or paste an image URL 🔗  
And get visually similar product results.

---

## 🚀 Tech Stack
- Python
- FastAPI
- Torch / TorchVision (ResNet50)
- FAISS
- HTML + CSS
- Uvicorn

---

## 🖼️ Features
- Search by **Image Upload**
- Search by **Image URL**
- Visual similarity using CNN embeddings
- Clean UI with CSS

---

## ⚙️ Setup Locally

```bash
git clone https://github.com/chandnikumarikumari/visual_product_matcher.git
cd visual_product_matcher
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python extract_features.py
uvicorn main:app --reload
