import cv2
import numpy as np
import glob
import os
from ultralytics import YOLO
from skimage.metrics import structural_similarity as ssim

# ── PATHS ──────────────────────────────────────────────────────────
MODEL_PATH     = r"E:\Dalnex\runs\detect\bottle_detector_v36\weights\best.pt"
REFERENCE_FILE = r"E:\Dalnex\reference_profile_v4.npy"
GOOD_DIR       = r"E:\Dalnex\good_bottles\*.jpg"
BAD_DIR        = r"E:\Dalnex\bad_bottles\*.jpg"

REF_W, REF_H = 100, 300

# ── LOAD MODEL & REFERENCE ─────────────────────────────────────────
model = YOLO(MODEL_PATH)

ref = np.load(REFERENCE_FILE, allow_pickle=True).item()
ref_mean = ref["mean"]
ref_std  = ref["std"]

# Convert reference for SSIM
ref_uint8 = (ref_mean * 255).astype(np.uint8)

# ── FEATURE EXTRACTION ─────────────────────────────────────────────
def extract_features(crop):
    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (REF_W, REF_H))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    edges   = cv2.Canny(blurred, 30, 100).astype(np.float32) / 255.0
    return edges

# ── DIFF SCORE ─────────────────────────────────────────────────────
def get_score(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    results = model(img, conf=0.25, verbose=False)[0]
    if len(results.boxes) == 0:
        return None

    best = max(results.boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = map(int, best.xyxy[0])

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    feat = extract_features(crop)
    diff = np.abs(feat - ref_mean) / (ref_std + 0.1)

    return float(np.mean(diff))

# ── SSIM SCORE ─────────────────────────────────────────────────────
def get_ssim(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    results = model(img, conf=0.25, verbose=False)[0]
    if len(results.boxes) == 0:
        return None

    best = max(results.boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = map(int, best.xyxy[0])

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (REF_W, REF_H))

    return float(ssim(resized, ref_uint8, data_range=255))

# ── SCORING DATASETS ───────────────────────────────────────────────
print("Scoring GOOD bottles...")
good_scores = []
for path in glob.glob(GOOD_DIR):
    s = get_score(path)
    if s is not None:
        good_scores.append(s)
        print(f"  GOOD {os.path.basename(path)}: {s:.4f}")

print("\nScoring BAD bottles...")
bad_scores = []
for path in glob.glob(BAD_DIR):
    s = get_score(path)
    if s is not None:
        bad_scores.append(s)
        print(f"  BAD  {os.path.basename(path)}: {s:.4f}")

# ── SUMMARY ────────────────────────────────────────────────────────
print("\n" + "="*50)

if good_scores:
    print(f"GOOD bottles ({len(good_scores)} images):")
    print(f"  min={min(good_scores):.4f}  max={max(good_scores):.4f}  mean={np.mean(good_scores):.4f}")

if bad_scores:
    print(f"\nBAD bottles ({len(bad_scores)} images):")
    print(f"  min={min(bad_scores):.4f}  max={max(bad_scores):.4f}  mean={np.mean(bad_scores):.4f}")

# ── FIND BEST DIFF THRESHOLD ───────────────────────────────────────
if good_scores and bad_scores:
    best_threshold = None
    best_accuracy  = 0

    for t in np.arange(0.05, 0.50, 0.005):
        correct_good = sum(s <= t for s in good_scores)
        correct_bad  = sum(s >  t for s in bad_scores)

        accuracy = (correct_good + correct_bad) / (len(good_scores) + len(bad_scores))

        if accuracy > best_accuracy:
            best_accuracy  = accuracy
            best_threshold = t

    print(f"\n{'='*50}")
    print(f"✅ Best DIFF threshold: {best_threshold:.3f}")
    print(f"✅ Accuracy:            {best_accuracy*100:.1f}%")

    print(f"Good correct: {sum(s <= best_threshold for s in good_scores)}/{len(good_scores)}")
    print(f"Bad  correct: {sum(s >  best_threshold for s in bad_scores)}/{len(bad_scores)}")

else:
    print("\n⚠ Not enough data to compute threshold")

# ── SSIM ANALYSIS ──────────────────────────────────────────────────
print("\n" + "="*50)
print("SSIM scores:")

good_ssim = [s for p in glob.glob(GOOD_DIR) if (s := get_ssim(p)) is not None]
bad_ssim  = [s for p in glob.glob(BAD_DIR)  if (s := get_ssim(p)) is not None]

if good_ssim:
    print(f"GOOD ssim: min={min(good_ssim):.3f} max={max(good_ssim):.3f} mean={np.mean(good_ssim):.3f}")

if bad_ssim:
    print(f"BAD  ssim: min={min(bad_ssim):.3f} max={max(bad_ssim):.3f} mean={np.mean(bad_ssim):.3f}")

# ── OPTIONAL: SUGGEST SSIM THRESHOLD ───────────────────────────────
if good_ssim and bad_ssim:
    best_t = None
    best_acc = 0

    for t in np.arange(0.5, 1.0, 0.01):
        correct_good = sum(s >= t for s in good_ssim)
        correct_bad  = sum(s <  t for s in bad_ssim)

        acc = (correct_good + correct_bad) / (len(good_ssim) + len(bad_ssim))

        if acc > best_acc:
            best_acc = acc
            best_t = t

    print(f"\nBest SSIM threshold: {best_t:.3f}")
    print(f"SSIM Accuracy:       {best_acc*100:.1f}%")