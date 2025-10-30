import cv2
import numpy as np
from ultralytics import YOLO
from keras.models import load_model
from collections import deque

# ======================================================
# CONFIGURATION
# ======================================================
VIDEO_PATH = "/home/bilal/Documents/FYP/videos/video_cropped.mp4"
OUTPUT_PATH = "/home/bilal/Documents/FYP/videos/output_action_cropped.mp4"

ACTION_CLASSES = ["looking_away", "raising_hand", "sleeping", "using_mobile", "writing_notes"]
FRAME_SMOOTHING = 5  # smooth predictions across frames

# Load YOLO model for student detection
print("🔍 Loading YOLO model...")
yolo = YOLO("yolov8n.pt")

# Load Action Classification model
print("🏃 Loading Action Model (.h5)...")
action_model = load_model("/home/bilal/Documents/FYP/my_action_classifier_final.h5")

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

print("🎥 Processing video for Action Detection...")

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

        # Preprocess crop for action model
        crop_resized = cv2.resize(crop, (224, 224))
        crop_normalized = crop_resized / 255.0
        crop_input = np.expand_dims(crop_normalized, axis=0)

        # Predict action
        preds = action_model.predict(crop_input, verbose=0)
        best_idx = np.argmax(preds)
        best_score = float(np.max(preds))
        best_label = ACTION_CLASSES[best_idx]

        # Smoothing
        recent_preds.append(best_label)
        smoothed_label = max(set(recent_preds), key=recent_preds.count)

        # Draw results
        label_text = f"{smoothed_label} ({best_score:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
        cv2.putText(frame, label_text, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)

    out.write(frame)

    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Action inference complete! Output saved to: {OUTPUT_PATH}")
