"""
Live commerce product recognition — YOLOv8 + CLIP
--------------------------------------------------
Run:
    python main.py --source video.mp4
    python main.py --source 0          # webcam
    python main.py --source rtmp://... # live stream

Requirements:
    pip install ultralytics transformers opencv-python torch Pillow
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel


# ── 1. Product catalog ────────────────────────────────────────────────────────
# Replace these with your actual products. Each entry is a text prompt that
# CLIP will match against. Be specific — "red Nike Air Max sneakers" works
# much better than just "shoes".
PRODUCT_CATALOG = [
    "red Nike Air Max sneakers",
    "white iPhone 15 smartphone",
    "blue denim jacket",
    "black leather handbag",
    "green matcha powder tin",
    "wireless Bluetooth earbuds in case",
    "pink facial serum bottle",
    "yellow sunglasses with oval frames",
]

# Confidence thresholds
YOLO_CONF = 0.4          # minimum YOLO detection confidence
CLIP_CONF = 0.22         # minimum CLIP similarity to show a label
FRAME_SKIP = 2           # process every Nth frame (speed vs accuracy tradeoff)


# ── 2. Model loader ───────────────────────────────────────────────────────────
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] Using device: {device}")

    print("[info] Loading YOLOv8...")
    yolo = YOLO("yolov8n.pt")   # 'n' = nano (fastest). Try 's' or 'm' for accuracy.

    print("[info] Loading CLIP...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    return yolo, clip_model, clip_processor, device


# ── 3. Pre-compute catalog embeddings (do once at startup) ────────────────────
def encode_catalog(catalog, clip_model, clip_processor, device):
    print("[info] Encoding product catalog...")
    inputs = clip_processor(text=catalog, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_out = clip_model.get_text_features(**inputs)
        if isinstance(text_out, torch.Tensor):
            text_embeds = text_out
        elif hasattr(text_out, "pooler_output") and text_out.pooler_output is not None:
            text_embeds = text_out.pooler_output
        elif hasattr(text_out, "last_hidden_state") and text_out.last_hidden_state is not None:
            # fall back to CLS token from last hidden state
            text_embeds = text_out.last_hidden_state[:, 0, :]
        elif isinstance(text_out, dict) and "text_embeds" in text_out:
            text_embeds = text_out["text_embeds"]
        else:
            raise RuntimeError("Unexpected output from CLIP text encoder")

        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    print(f"[info] Catalog ready: {len(catalog)} products")
    return text_embeds


# ── 4. CLIP matching ──────────────────────────────────────────────────────────
def match_product(crop_bgr, clip_model, clip_processor, catalog_embeds, catalog, device):
    """Given a BGR crop from OpenCV, return (product_name, similarity_score)."""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)

    inputs = clip_processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        img_out = clip_model.get_image_features(**inputs)
        if isinstance(img_out, torch.Tensor):
            img_embed = img_out
        elif hasattr(img_out, "pooler_output") and img_out.pooler_output is not None:
            img_embed = img_out.pooler_output
        elif hasattr(img_out, "last_hidden_state") and img_out.last_hidden_state is not None:
            img_embed = img_out.last_hidden_state[:, 0, :]
        elif isinstance(img_out, dict) and "image_embeds" in img_out:
            img_embed = img_out["image_embeds"]
        else:
            raise RuntimeError("Unexpected output from CLIP image encoder")

        img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)

    similarities = (img_embed @ catalog_embeds.T).squeeze(0)
    best_idx = similarities.argmax().item()
    best_score = similarities[best_idx].item()

    return catalog[best_idx], best_score


# ── 5. Drawing helpers ────────────────────────────────────────────────────────
COLORS = [
    (52, 211, 153),   # teal
    (167, 139, 250),  # purple
    (251, 146, 60),   # orange
    (96, 165, 250),   # blue
    (244, 114, 182),  # pink
]

def draw_box(frame, x1, y1, x2, y2, label, score, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"{label}  {score:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    ty = max(y1 - 8, th + 4)

    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 8, ty + 2), color, -1)
    cv2.putText(frame, text, (x1 + 4, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_fps(frame, fps):
    cv2.putText(frame, f"{fps:.1f} fps", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)


# ── 6. Main loop ──────────────────────────────────────────────────────────────
def run(source):
    yolo, clip_model, clip_processor, device = load_models()
    catalog_embeds = encode_catalog(PRODUCT_CATALOG, clip_model, clip_processor, device)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    frame_idx = 0
    last_boxes = []   # cache detections across skipped frames
    t_prev = time.time()

    print("[info] Starting — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % FRAME_SKIP == 0:
            last_boxes = []

            # ── YOLOv8 detection ───────────────────────────────────────
            results = yolo(frame, conf=YOLO_CONF, verbose=False)[0]

            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Guard against degenerate boxes
                if x2 - x1 < 20 or y2 - y1 < 20:
                    continue

                crop = frame[y1:y2, x1:x2]

                # ── CLIP matching ──────────────────────────────────────
                name, score = match_product(
                    crop, clip_model, clip_processor,
                    catalog_embeds, PRODUCT_CATALOG, device
                )

                if score >= CLIP_CONF:
                    color = COLORS[i % len(COLORS)]
                    last_boxes.append((x1, y1, x2, y2, name, score, color))

        # Draw cached boxes on every frame (smooth even when skipping)
        for x1, y1, x2, y2, name, score, color in last_boxes:
            draw_box(frame, x1, y1, x2, y2, name, score, color)

        # FPS counter
        t_now = time.time()
        fps = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now
        draw_fps(frame, fps)

        cv2.imshow("Live product recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[info] Done")


# ── 7. Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live product recognition")
    parser.add_argument(
        "--source", default="0",
        help="Video source: path to mp4, '0' for webcam, or RTMP URL"
    )
    args = parser.parse_args()

    # Convert '0' string to int for webcam
    source = int(args.source) if args.source.isdigit() else args.source
    run(source)
