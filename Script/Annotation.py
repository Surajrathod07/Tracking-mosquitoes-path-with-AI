import cv2
import numpy as np

# ─── CONFIG ───────────────────────────────────────────────────────────
VIDEO_PATH  = r"C:\Users\Suraj Rathod\Documents\MosquitoProject\Input-023.mp4"
OUTPUT_PATH = r"C:\Users\Suraj Rathod\Documents\MosquitoProject\output_mosquito_blob_detected.mp4"

# Motion threshold (frame-diff)
MOTION_THR  = 5       # very sensitive to even tiny movement

# Blob detector parameters
MIN_AREA     = 1      # allow blobs as small as 1 px
MAX_AREA     = 200    # ignore anything much larger
BRIGHT_THR   = 150    # catch even moderately bright spots
BLOB_STEP    = 5      # step between thresholds

# Morphology
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

# ─── SETUP VIDEO I/O & BACKGROUND ─────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (W, H))

# ─── CONFIGURE BLOB DETECTOR ──────────────────────────────────────────
params = cv2.SimpleBlobDetector_Params()
params.minThreshold = BRIGHT_THR
params.maxThreshold = 255
params.thresholdStep = BLOB_STEP

params.filterByArea = True
params.minArea = MIN_AREA
params.maxArea = MAX_AREA

params.filterByCircularity = False
params.filterByConvexity   = False
params.filterByInertia     = False

detector = cv2.SimpleBlobDetector_create(params)

print(f"[INFO] Starting blob-based mosquito detection...")

# ─── PRIME PREVIOUS FRAME ─────────────────────────────────────────────
ret, prev = cap.read()
if not ret:
    raise RuntimeError("Unable to read first frame.")
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

frame_i = 1
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1) Motion mask: very low threshold
    diff = cv2.absdiff(gray, prev_gray)
    _, motion_mask = cv2.threshold(diff, MOTION_THR, 255, cv2.THRESH_BINARY)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, KERNEL)

    # 2) Detect bright blobs on the gray image
    keypoints = detector.detect(gray)

    # 3) Draw only those blobs that also moved this frame
    annotated = frame.copy()
    for kp in keypoints:
        x, y = map(int, kp.pt)
        r    = int(kp.size // 2) + 1
        # check motion in the blob’s local area
        x1, y1 = max(x-r, 0), max(y-r, 0)
        x2, y2 = min(x+r, W-1), min(y+r, H-1)
        patch = motion_mask[y1:y2, x1:x2]
        if np.any(patch):  # if any pixel moved
            # draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(annotated, "Mosquito", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1)

    out.write(annotated)
    prev_gray = gray
    if frame_i % 20 == 0:
        print(f"[FRAME {frame_i}] done, blobs detected: {len(keypoints)}")
    frame_i += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\n✅ Done! Output video:\n  {OUTPUT_PATH}")
