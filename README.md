<p align="center">
  <img src="https://img.shields.io/badge/YOLOv8-Object%20Detection-00C853?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/CLIP-Zero--Shot%20Matching-7C3AED?style=for-the-badge&logo=openai&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-Video%20Pipeline-E53935?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EF4444?style=for-the-badge&logo=pytorch&logoColor=white"/>
</p>

<h1 align="center"> Live Commerce Product Recognition</h1>

<p align="center">
  Real-time AI system that detects and identifies products in live shopping streams using YOLOv8 + CLIP.
  <br/>
  Inspired by Taobao Live and Douyin commerce pipelines.
</p>

---

#  Overview

This project recreates the core AI pipeline used in modern live commerce platforms such as Taobao Live, Douyin Shop, and JD.com.

Given a live stream, webcam feed, or recorded video, the system automatically:

- Detects products in real time using YOLOv8
- Identifies products using CLIP zero-shot matching
- Matches products against a text catalog
- Draws bounding boxes and labels directly on video frames

Unlike traditional image classifiers, this project requires:

 No labeled dataset  
 No custom training  
 No retraining when products change  

You simply describe products using natural language.

Example:

```python
"Dior Saddle Bag in brown leather"
```

CLIP handles the recognition automatically.

---

#  Features

- Real-time product detection
- Zero-shot product recognition
- Webcam / MP4 / RTMP stream support
- GPU acceleration with PyTorch
- OpenCV video pipeline
- Portfolio-ready AI project
- Inspired by real ecommerce AI systems
- Easily extendable for live commerce applications

---

#  AI Pipeline

```text
Video Stream
     │
     ▼
YOLOv8 Object Detection
     │
     ▼
Crop Product Regions
     │
     ▼
CLIP Image Encoder
     │
     ▼
Cosine Similarity Matching
     │
     ▼
Best Product Label
     │
     ▼
Annotated Video Output
```

---

#  System Architecture

```text
┌────────────────────────────────────────────────────────────┐
│                      VIDEO INPUT                           │
│           webcam / mp4 / RTMP stream                       │
└───────────────────────┬────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────┐
│                    YOLOv8 DETECTOR                         │
│    Bounding boxes + confidence prediction                  │
└───────────────────────┬────────────────────────────────────┘
                        │
                        ▼
                Crop Product Regions
                        │
                        ▼
┌────────────────────────────────────────────────────────────┐
│                     CLIP ENCODER                           │
│   Image embeddings ↔ Text embeddings                       │
└───────────────────────┬────────────────────────────────────┘
                        │
                        ▼
               Cosine Similarity Search
                        │
                        ▼
┌────────────────────────────────────────────────────────────┐
│                  PRODUCT IDENTIFICATION                    │
│      label + similarity score overlay                      │
└────────────────────────────────────────────────────────────┘
```

---

# Core Technologies

## 1. YOLOv8 — Real-Time Object Detection

YOLOv8 is a state-of-the-art object detection model optimized for real-time inference.

### Key Concepts

| Concept | Explanation |
|---|---|
| Single-shot detection | Detects all objects in one forward pass |
| Anchor-free architecture | Predicts boxes directly without anchor templates |
| Non-Max Suppression | Removes duplicate overlapping boxes |
| Multi-scale feature extraction | Detects both small and large objects |
| Real-time inference | Achieves high FPS on GPU |

### Why YOLOv8?

- Extremely fast
- Industry standard for live AI systems
- GPU optimized
- Excellent speed/accuracy tradeoff

---

## 2. CLIP — Zero-Shot Product Recognition

CLIP aligns images and text into the same embedding space.

This enables matching products using only text descriptions.

### Example Catalog

```python
PRODUCT_CATALOG = [
    "red Nike Air Max sneakers",
    "Dior Saddle Bag in brown leather",
    "Samsung Galaxy S24 Ultra smartphone"
]
```

The model compares image embeddings against text embeddings using cosine similarity.

No retraining required.

---

## 3. Vision Transformer (ViT)

CLIP uses a Vision Transformer as its image encoder.

### Process

```text
Input Image
     ↓
Split into patches
     ↓
Transformer self-attention
     ↓
512-dimensional embedding
```

Self-attention allows the model to understand:

- logos
- packaging
- colors
- textures
- product context

---

## 4. Embedding Similarity

The recognition step relies on cosine similarity between vectors.

```python
img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

similarities = img_embed @ catalog_embeds.T

best_idx = similarities.argmax()
```

### Why normalization?

L2 normalization projects embeddings onto a unit sphere, making cosine similarity stable and comparable.

---

#  Performance Optimizations

| Optimization | Benefit |
|---|---|
| Frame skipping | Reduces compute load |
| Precomputed text embeddings | Faster inference |
| GPU acceleration | Major speed improvement |
| `torch.no_grad()` | Lower memory usage |
| Batched similarity search | Efficient catalog matching |

---

# 🛠️ Tech Stack

| Technology | Role |
|---|---|
| YOLOv8 | Object detection |
| CLIP | Zero-shot recognition |
| PyTorch | Deep learning inference |
| OpenCV | Video processing |
| Pillow | Image conversion |
| Transformers | CLIP loading |

---

# 📂 Project Structure

```text
live_product_recognition/
│
├── main.py
├── process_video.py
├── requirements.txt
└── README.md
```

---

#  Installation

## 1. Clone Repository

```bash
git clone https://github.com/your-username/live-product-recognition.git
cd live-product-recognition
```

## 2. Install Dependencies

```bash
pip install ultralytics transformers opencv-python torch Pillow
```

---

#  Usage

## Webcam

```bash
python main.py --source 0
```

## Video File

```bash
python main.py --source stream.mp4
```

## RTMP Stream

```bash
python main.py --source rtmp://your-stream-url
```

## Export Annotated Video

```bash
python process_video.py --input stream.mp4 --output demo.mp4
```

---

# ⚙️ Configuration

| Parameter | Default | Description |
|---|---|---|
| YOLO_CONF | 0.4 | Detection confidence threshold |
| CLIP_CONF | 0.22 | Product matching confidence |
| FRAME_SKIP | 2 | Detection every N frames |
| YOLO Model | yolov8n | Detection model size |

---



---

#  Future Improvements

| Feature | Description |
|---|---|
| Streamlit UI | Upload and process videos in browser |
| FAISS Search | Scale to massive catalogs |
| Fine-tuned CLIP | Improve domain-specific accuracy |
| TensorRT Export | Faster GPU inference |
| RTMP Ingest Server | Real live-stream deployment |
| Ecommerce Integration | Display product metadata and pricing |

---

#  Real-World Applications

- Live shopping platforms
- Ecommerce AI systems
- Interactive livestreams
- Automated product tagging
- Smart retail analytics
- AI-powered visual search

---

#  References

- YOLOv8 — Ultralytics
- OpenAI CLIP
- Vision Transformer (ViT)
- PyTorch
- OpenCV

---

#  Author

Built as a computer vision portfolio project inspired by modern China ecommerce AI systems such as:

- Taobao Live
- Douyin Shop
- JD.com Live Commerce

---

<p align="center">
  ⭐ If you like this project, give it a star on GitHub!
</p>
