# Live commerce product recognition

Real-time product detection and identification in live shopping streams, using YOLOv8 for object detection and CLIP for zero-shot product matching.

## How it works

1. **YOLOv8** detects all objects in each frame and returns bounding boxes
2. Each crop is passed to **CLIP**, which computes an image embedding
3. CLIP compares the image embedding against pre-computed text embeddings for every product in your catalog
4. The closest match (by cosine similarity) is displayed as an overlay

No training required — just update the product catalog text descriptions.

## Setup

```bash
pip install ultralytics transformers opencv-python torch Pillow
```

## Usage

```bash
# From a video file
python main.py --source video.mp4

# From webcam
python main.py --source 0

# From an RTMP live stream (e.g. Douyin/TikTok live)
python main.py --source rtmp://your-stream-url

# Save annotated output to a file
python process_video.py --input input.mp4 --output annotated.mp4
```

## Customising the product catalog

Edit `PRODUCT_CATALOG` in `main.py`. Be specific in your descriptions:

```python
PRODUCT_CATALOG = [
    "red Nike Air Max 90 sneakers",   # Good — specific
    "shoes",                           # Bad — too vague
    "Dior Saddle Bag in brown leather",
    "Samsung Galaxy S24 Ultra in titanium black",
]
```

CLIP is zero-shot, meaning it can recognise products it has never been fine-tuned on. More descriptive prompts = better accuracy.

## Tuning performance

| Parameter | File | Effect |
|-----------|------|--------|
| `YOLO_CONF` | main.py | Lower = more detections, more false positives |
| `CLIP_CONF` | main.py | Lower = more labels shown, less reliable |
| `FRAME_SKIP` | main.py | Higher = faster, less responsive |
| YOLO model size | main.py | `yolov8n` (fast) → `yolov8s` → `yolov8m` (accurate) |

## CV project tips

- Record a short demo on a Douyin-style live stream video (or simulate one)
- Export the annotated video with `process_video.py` to include in your portfolio
- Add a Streamlit web UI for uploading videos and getting annotated output
- Extend it: scrape product prices from Taobao and show them in the overlay

## Stack

- [YOLOv8](https://github.com/ultralytics/ultralytics) — object detection
- [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) — zero-shot image-text matching
- OpenCV — video I/O and rendering
- PyTorch — model inference
