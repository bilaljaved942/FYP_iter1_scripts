import torch
import cv2
import numpy as np
from ultralytics import YOLO
import clip
from PIL import Image
from collections import deque

# ======================================================
# CONFIGURATION
# ======================================================
VIDEO_PATH = "/home/bilal/Documents/FYP/videos/video_cropped.mp4"
OUTPUT_PATH = "/home/bilal/Documents/FYP/videos/output_emotion_clip_v2.mp4"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {DEVICE}")

# ======================================================
# 1️⃣ Emotion Classes with Optimized CLIP Prompts
# ======================================================
CLIP_EMOTIONS = {
    "confusion": [
        "a person looking confused",
        "a person scratching their head in confusion",
        "a person frowning and thinking hard",
        "a person with a puzzled face expression"
    ],
    "happy": [
        "a person smiling happily",
        "a person laughing with joy",
        "a person with a big smile on their face"
    ],
    "neutral": [
        "a person sitting still with a neutral face",
        "a person showing no emotion and looking forward",
        "a person with a blank facial expression"
    ],
    "yawning": [
        "a person yawning with mouth open",
        "a person feeling sleepy with eyes half closed",
        "a person covering their mouth while yawning"
    ]
}

# ======================================================
# 2️⃣ Load YOLO Model (Person Detection)
# ======================================================
print("🔍 Loading YOLO model for person detection...")
yolo = YOLO("yolov8n.pt")

# ======================================================
# 3️⃣ Load CLIP Model
# ======================================================
print("💬 Loading CLIP model...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE, jit=False)
clip_model.eval()

# Compute averaged text embeddings per emotion
print("🧠 Creating emotion text embeddings...")
text_embeddings = []
for emotion, prompts in CLIP_EMOTIONS.items():
    tokens = clip.tokenize(prompts).to(DEVICE)
    with torch.no_grad():
        emb = clip_model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        emb = emb.mean(dim=0, keepdim=True)
    text_embeddings.append(emb)

text_embeddings = torch.cat(text_embeddings, dim=0)
EMOTION_CLASSES = list(CLIP_EMOTIONS.keys())
print(f"✅ CLIP model ready with {len(EMOTION_CLASSES)} emotion classes.")

# ======================================================
# 4️⃣ Video Reader & Writer
# ======================================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError("❌ Cannot open video file!")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print("🎥 Processing video...")

# ======================================================
# 5️⃣ Frame-by-Frame CLIP Emotion Detection
# ======================================================
frame_count = 0
recent_preds = deque(maxlen=5)  # to smooth predictions over frames

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Detect persons using YOLO
    results = yolo(frame, classes=[0], conf=0.5, verbose=False)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Convert to PIL image for CLIP
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        with torch.no_grad():
            img_tensor = clip_preprocess(pil_img).unsqueeze(0).to(DEVICE)
            img_emb = clip_model.encode_image(img_tensor)
            img_emb /= img_emb.norm(dim=-1, keepdim=True)

            # Compute similarity
            similarities = (img_emb @ text_embeddings.T).squeeze(0)
            best_idx = similarities.argmax().item()
            best_score = similarities[best_idx].item()
            second_best = torch.topk(similarities, 2).values[-1].item()

            # Balanced fallback logic
            if best_score < 0.20 or (best_score - second_best) < 0.015:
                best_label = "neutral"
            else:
                best_label = EMOTION_CLASSES[best_idx]

        # Add prediction to smoothing queue
        recent_preds.append(best_label)
        smoothed_label = max(set(recent_preds), key=recent_preds.count)

        # Visualization
        color = (0, 255, 0)
        label_text = f"{smoothed_label} ({best_score:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label_text, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    out.write(frame)

    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

# ======================================================
# 6️⃣ Cleanup
# ======================================================
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Emotion CLIP Inference complete! Output saved to: {OUTPUT_PATH}")
