

import argparse
import time
import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPModel, CLIPProcessor
from main import (
    PRODUCT_CATALOG, YOLO_CONF, CLIP_CONF, FRAME_SKIP,
    load_models, encode_catalog, match_product, draw_box, COLORS
)


def process_video(input_path, output_path):
    yolo, clip_model, clip_processor, device = load_models()
    catalog_embeds = encode_catalog(PRODUCT_CATALOG, clip_model, clip_processor, device)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx, last_boxes = 0, []
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % FRAME_SKIP == 0:
            last_boxes = []
            results = yolo(frame, conf=YOLO_CONF, verbose=False)[0]
            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if x2 - x1 < 20 or y2 - y1 < 20:
                    continue
                crop = frame[y1:y2, x1:x2]
                name, score = match_product(crop, clip_model, clip_processor,
                                            catalog_embeds, PRODUCT_CATALOG, device)
                if score >= CLIP_CONF:
                    last_boxes.append((x1, y1, x2, y2, name, score, COLORS[i % len(COLORS)]))

        for x1, y1, x2, y2, name, score, color in last_boxes:
            draw_box(frame, x1, y1, x2, y2, name, score, color)

        out.write(frame)

        if frame_idx % 30 == 0:
            elapsed = time.time() - t0
            print(f"  Frame {frame_idx}/{total}  ({elapsed:.0f}s elapsed)")

    cap.release()
    out.release()
    print(f"[done] Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="annotated_output.mp4")
    args = parser.parse_args()
    process_video(args.input, args.output)
