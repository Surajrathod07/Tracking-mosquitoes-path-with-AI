
import cv2
import numpy as np
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = r"C:\mosquito project\mosquito-dataset\mosquito-runs\yolov8_mosquito2\weights\best100.pt"
VIDEO_PATH = r"Input_video-path"
OUTPUT_PATH = r"output_mosquito__detected.mp4"

YOLO_CONF = 1.0  # Detect even uncertain blobs
YOLO_IOU = 0.4

# --- Load model ---
model = YOLO(MODEL_PATH)

# --- Read video ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(OUTPUT_PATH,
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (width, height))

print(f"[INFO] Video loaded ({total_frames} frames at {fps:.2f} fps)")
print("[INFO] Starting mosquito detection...")

# --- Prepare motion analysis ---
prev_gray = None
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)

frame_index = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Step 1: Sharpen blurred frame
    sharp = cv2.filter2D(frame, -1, sharpen_kernel)

    # Step 2: Compute motion mask (pixel-wise difference)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is not None:
        diff = cv2.absdiff(prev_gray, gray)
        _, motion_mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
        fused = cv2.addWeighted(sharp, 0.8, motion_mask, 0.2, 0)
    else:
        fused = sharp

    prev_gray = gray.copy()

    # Step 3: YOLOv8 Detection (very low conf for tiny motion)
    res = model.predict(source=fused, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)[0]
    annotated = res.plot()

    # Step 4: Output video frame
    out.write(annotated)
    frame_index += 1

    if frame_index % 10 == 0:
        print(f"[FRAME {frame_index}] processed")

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"[✔] Detection complete. Output saved to:\n{OUTPUT_PATH}")