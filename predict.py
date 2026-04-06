import cv2
import numpy as np
from ultralytics import YOLO

# ── PATHS ──────────────────────────────────────────────────────────
MODEL_PATH = r"E:\Dalnex\runs\detect\bottle_detector_v36\weights\best.pt"

REFERENCE_FRONT = r"E:\Dalnex\reference_profile_front.npy"
REFERENCE_SIDE  = r"E:\Dalnex\reference_profile_side.npy"

# ── PARAMETERS ─────────────────────────────────────────────────────
REF_W, REF_H = 100, 300

EDGE_THRESHOLD  = 0.135
CAP_ANGLE_LIMIT = 18.0

SIDE_RATIO_MAX  = 0.32

# ── LOAD MODEL ─────────────────────────────────────────────────────
model = YOLO(MODEL_PATH)

# ── LOAD REFERENCE ─────────────────────────────────────────────────
def _load_ref(path):
    raw = np.load(path, allow_pickle=True)

    if raw.ndim == 0:
        d = raw.item()
        return {"mean": d["mean"], "std": d["std"]}

    return {
        "mean": raw.astype(np.float32),
        "std": np.ones_like(raw) * 0.1
    }

ref_front = _load_ref(REFERENCE_FRONT)
ref_side  = _load_ref(REFERENCE_SIDE)

# ── PICK REFERENCE ─────────────────────────────────────────────────
def _pick_ref(crop):
    h, w = crop.shape[:2]
    ratio = w / h if h > 0 else 1.0

    if ratio <= SIDE_RATIO_MAX:
        return ref_side, "side"
    return ref_front, "front"

# ── EDGE EXTRACTION ────────────────────────────────────────────────
def extract_edges(crop):
    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (REF_W, REF_H))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    edges   = cv2.Canny(blurred, 30, 100).astype(np.float32) / 255.0
    return edges

# ── EDGE SCORE ─────────────────────────────────────────────────────
def edge_score(crop, ref):
    edges = extract_edges(crop)
    diff  = np.abs(edges - ref["mean"]) / (ref["std"] + 0.1)
    return float(np.mean(diff))

# ── CAP CHECK (placeholder) ────────────────────────────────────────
def cap_angle_check(crop):
    return False, 0.0

# ── DEBUG VISUALIZATION ────────────────────────────────────────────
def debug_view(crop, ref, e_score, label):
    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (REF_W, REF_H))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    edges   = cv2.Canny(blurred, 30, 100)

    ref_img = (ref["mean"] * 255).astype(np.uint8)

    edge_f = edges.astype(np.float32) / 255.0
    diff   = np.abs(edge_f - ref["mean"]) / (ref["std"] + 0.1)
    diff_u8 = np.clip(diff * 255, 0, 255).astype(np.uint8)
    heat   = cv2.applyColorMap(diff_u8, cv2.COLORMAP_PLASMA)

    edges_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    ref_vis   = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)

    combined = np.hstack([
        cv2.resize(crop, (150, 300)),
        cv2.resize(edges_vis, (150, 300)),
        cv2.resize(ref_vis, (150, 300)),
        cv2.resize(heat, (150, 300))
    ])

    cv2.putText(combined, f"{label} | score={e_score:.3f}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255,255,255), 1)

    cv2.imshow("DEBUG VIEW", combined)

# ── FINAL SCORING ──────────────────────────────────────────────────
def score_bottle(crop):
    ref, bottle_type = _pick_ref(crop)

    e_score = edge_score(crop, ref)
    cap_bad, cap_dev = cap_angle_check(crop)

    edge_bad = e_score > EDGE_THRESHOLD
    is_bad   = edge_bad or cap_bad

    reasons = []
    if edge_bad:
        reasons.append(f"edge={e_score:.3f}>{EDGE_THRESHOLD}")
    if cap_bad:
        reasons.append(f"cap_angle={cap_dev:.1f}>{CAP_ANGLE_LIMIT}")

    reason = " | ".join(reasons) if reasons else "OK"

    return is_bad, e_score, cap_dev, reason, bottle_type, ref

# ── MAIN ANALYSIS ──────────────────────────────────────────────────
def analyze(img_path, debug=False):
    img = cv2.imread(img_path)
    if img is None:
        print("Cannot load image.")
        return

    results = model(img, conf=0.25, verbose=False)[0]
    output  = img.copy()

    if len(results.boxes) == 0:
        print("No bottle detected.")
        return

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        is_bad, e_score, cap_dev, reason, btype, ref = score_bottle(crop)

        color = (0, 0, 255) if is_bad else (0, 200, 0)
        label = "BAD" if is_bad else "GOOD"

        print(f"\nBottle {i+1} [{btype}] → {label}")
        print(f"  edge={e_score:.3f}  cap={cap_dev:.1f}°")
        print(f"  reason: {reason}")

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)

        cv2.putText(output, label,
                    (x1, max(y1 - 10, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2)

        cv2.putText(output,
                    f"e={e_score:.2f} cap={cap_dev:.1f}",
                    (x1, max(y1 - 35, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1)

        # ✅ DEBUG MODE
        if debug:
            debug_view(crop, ref, e_score, label)

    cv2.imshow("Result", cv2.resize(output, (512, 640)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ── RUN LOOP ───────────────────────────────────────────────────────
if __name__ == "__main__":
    debug_mode = input("Enable debug view? (y/n): ").strip().lower() == "y"

    while True:
        path = input("\nImage path (or 'q' to quit): ").strip().strip('"')

        if path.lower() == "q":
            break

        analyze(path, debug=debug_mode)