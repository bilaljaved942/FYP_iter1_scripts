import torch
import cv2
import numpy as np
from ultralytics import YOLO
import clip
from PIL import Image

# ======================================================
# CONFIGURATION
# ======================================================
VIDEO_PATH = "/home/bilal/Documents/FYP/videos/video_cropped.mp4"
OUTPUT_PATH = "/home/bilal/Documents/FYP/videos/output_action_clip_refined_v3.mp4"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {DEVICE}")

# ======================================================
# 1️⃣ Define CLIP Labels with Descriptive Prompts
# ======================================================
CLIP_LABELS = {
    "using_mobile": [
        "a person holding a phone in their hand",
        "a person looking down at a mobile phone",
        "a person texting on a smartphone"
    ],
    "writing_notes": [
        "a person writing something in a notebook",
        "a person holding a pen and writing on paper",
        "a person sitting and writing notes in a notebook",
        "a person holding a pen near a notebook and writing"
    ],
    "raising_hand": [
        "a person raising one hand in the air",
        "a person with one arm lifted up",
        "a person stretching their hand upward"
    ],
    "sleeping": [
        "a person sleeping with eyes closed",
        "a person resting their head on a desk",
        "a person dozing off with head down"
    ],
    "looking_away": [
        "a person turning their head to the right",
        "a person turning their head to the left",
        "a person facing away from the camera"
    ],
    "neutral": [
        "a person sitting still and looking straight",
        "a person sitting upright with no movement",
        "a person facing forward doing nothing"
    ]
}

# ======================================================
# 2️⃣ Load YOLO Model
# ======================================================
print("🔍 Loading YOLO model for person detection...")
yolo = YOLO("yolov8n.pt")

# ======================================================
# 3️⃣ Load CLIP Model
# ======================================================
print("💬 Loading CLIP model...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE, jit=False)
clip_model.eval()

# Compute averaged text embeddings per label
print("🧠 Creating text embeddings from prompts...")
text_embeddings = []
for label, prompts in CLIP_LABELS.items():
    tokens = clip.tokenize(prompts).to(DEVICE)
    with torch.no_grad():
        emb = clip_model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        emb = emb.mean(dim=0, keepdim=True)  # average prompts per label
    text_embeddings.append(emb)
text_embeddings = torch.cat(text_embeddings, dim=0)
CLIP_CLASSES = list(CLIP_LABELS.keys())
print(f"✅ CLIP model ready with {len(CLIP_CLASSES)} classes.")

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
# 5️⃣ Frame-by-Frame CLIP Inference
# ======================================================
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Detect persons
    results = yolo(frame, classes=[0], conf=0.5, verbose=False)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Convert crop to PIL for CLIP
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        with torch.no_grad():
            clip_img = clip_preprocess(pil_img).unsqueeze(0).to(DEVICE)
            img_emb = clip_model.encode_image(clip_img)
            img_emb /= img_emb.norm(dim=-1, keepdim=True)

            # Compute cosine similarity
            similarities = (img_emb @ text_embeddings.T).squeeze(0)
            best_idx = similarities.argmax().item()
            best_score = similarities[best_idx].item()
            second_best = torch.topk(similarities, 2).values[-1].item()

            # Revised neutral fallback (lower threshold for neutral)
            if best_score < 0.18 or (best_score - second_best) < 0.015:
                best_label = "neutral"
            else:
                best_label = CLIP_CLASSES[best_idx]

        # Visualization
        color = (0, 255, 0)
        label_text = f"{best_label.replace('_', ' ')} ({best_score:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label_text, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    out.write(frame)

    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

# ======================================================
# 6️⃣ Cleanup
# ======================================================
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Inference complete! Output saved to: {OUTPUT_PATH}")
