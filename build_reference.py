# build_reference.py — builds BOTH front and side references in one pass
import cv2, numpy as np, glob, os
from ultralytics import YOLO

MODEL_PATH         = r"E:\Dalnex\runs\detect\bottle_detector_v36\weights\best.pt"
GOOD_IMAGES_DIR    = r"E:\Dalnex\good_bottles\*.jpg"

OUTPUT_FRONT       = r"E:\Dalnex\reference_profile_front.npy"
OUTPUT_SIDE        = r"E:\Dalnex\reference_profile_side.npy"

REF_W, REF_H       = 100, 300
SIDE_RATIO_MAX     = 0.32   # ratio below this → side view
FRONT_RATIO_MIN    = 0.35   # ratio above this → front view

model = YOLO(MODEL_PATH)

def extract_features(crop):
    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (REF_W, REF_H))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    edges   = cv2.Canny(blurred, 30, 100).astype(np.float32) / 255.0
    return edges

images = glob.glob(GOOD_IMAGES_DIR)
print(f"Found {len(images)} images\n")

front_stack = []
side_stack  = []

for path in images:
    img = cv2.imread(path)
    if img is None:
        continue

    results = model(img, conf=0.25, verbose=False)[0]
    if len(results.boxes) == 0:
        print(f"  ⚠ No detection: {os.path.basename(path)}")
        continue

    best = max(results.boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = map(int, best.xyxy[0])
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        continue

    ch, cw = crop.shape[:2]
    if cw > ch:
        print(f"  ⚠ Skipping landscape crop {cw}x{ch}: {os.path.basename(path)}")
        continue

    ratio = cw / ch

    if ratio >= FRONT_RATIO_MIN:
        front_stack.append(extract_features(crop))
        print(f"  [FRONT] ✓ {os.path.basename(path)}  {cw}x{ch}  ratio={ratio:.2f}")
    elif ratio <= SIDE_RATIO_MAX:
        side_stack.append(extract_features(crop))
        print(f"  [SIDE]  ✓ {os.path.basename(path)}  {cw}x{ch}  ratio={ratio:.2f}")
    else:
        print(f"  [????]  ⚠ Ambiguous ratio={ratio:.2f}, skipping: {os.path.basename(path)}")

print(f"\nFront crops: {len(front_stack)}   Side crops: {len(side_stack)}")

def save_ref(stack, output_path, label):
    if len(stack) < 3:
        print(f"\n❌ {label}: only {len(stack)} crops — need at least 3. Skipping.")
        return
    mean = np.mean(stack, axis=0)
    std  = np.std(stack, axis=0)
    np.save(output_path, {"mean": mean, "std": std})
    print(f"✅ {label} reference saved → {output_path}")
    preview = cv2.resize((mean * 255).astype(np.uint8), (200, 600))
    cv2.imshow(f"{label} reference edge map", preview)

save_ref(front_stack, OUTPUT_FRONT, "FRONT")
save_ref(side_stack,  OUTPUT_SIDE,  "SIDE")

if front_stack or side_stack:
    cv2.waitKey(0)
    cv2.destroyAllWindows()