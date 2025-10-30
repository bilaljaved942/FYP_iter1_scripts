import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from collections import deque

# ======================================================
# CONFIGURATION
# ======================================================
VIDEO_PATH = "/home/bilal/Documents/FYP/videos/video_cropped.mp4"
OUTPUT_PATH = "/home/bilal/Documents/FYP/videos/output_emotion_model.mp4"

EMOTION_CLASSES = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
FRAME_SMOOTHING = 5  # average predictions across frames

# Load YOLO model for person detection
print("🔍 Loading YOLO model...")
yolo = YOLO("yolov8n.pt")

# Load emotion model
print("💬 Loading Emotion Model (.keras)...")
emotion_model = load_model("/home/bilal/Documents/FYP/models/emotion_model.keras")

# ======================================================
# VIDEO READER & WRITER
# ======================================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError("❌ Cannot open video file!")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print("🎥 Processing video for Emotion Detection...")

recent_preds = deque(maxlen=FRAME_SMOOTHING)
frame_count = 0

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

        # Preprocess for Emotion model
        crop_resized = cv2.resize(crop, (224, 224))
        crop_normalized = crop_resized / 255.0
        crop_input = np.expand_dims(crop_normalized, axis=0)

        # Prediction
        preds = emotion_model.predict(crop_input, verbose=0)
        best_idx = np.argmax(preds)
        best_score = float(np.max(preds))
        best_label = EMOTION_CLASSES[best_idx]

        # Smoothing
        recent_preds.append(best_label)
        smoothed_label = max(set(recent_preds), key=recent_preds.count)

        # Visualization
        label_text = f"{smoothed_label} ({best_score:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    out.write(frame)

    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Emotion inference complete! Output saved to: {OUTPUT_PATH}")
