# debug.py
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH     = r"E:\Dalnex\runs\detect\bottle_detector_v36\weights\best.pt"
REFERENCE_FILE = r"E:\Dalnex\reference_profile_v4.npy"
REF_W, REF_H   = 100, 300
MIN_RATIO       = 0.35
EDGE_THRESHOLD  = 0.135

model    = YOLO(MODEL_PATH)
ref      = np.load(REFERENCE_FILE, allow_pickle=True).item()
ref_mean = ref["mean"]
ref_std  = ref["std"]

# ── helpers ───────────────────────────────────────────────────────────────────

def get_crop(img_path):
    """Run YOLO on an image, return (crop, score, label) or None on failure."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"  ✗ Cannot load: {img_path}")
        return None, None, None

    results = model(img, conf=0.25, verbose=False)[0]
    if len(results.boxes) == 0:
        print(f"  ✗ No detection: {img_path}")
        return None, None, None

    best = max(results.boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = map(int, best.xyxy[0])
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None, None

    ch, cw = crop.shape[:2]
    ratio = cw / ch if ch > 0 else 1.0
    if ratio < MIN_RATIO or cw > ch:
        print(f"  ⚠ Bad crop {cw}x{ch} ratio={ratio:.2f} — YOLO missed the bottle")
        return None, None, None

    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (REF_W, REF_H))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    edges   = cv2.Canny(blurred, 30, 100).astype(np.float32) / 255.0
    diff    = np.abs(edges - ref_mean) / (ref_std + 0.1)
    score   = float(np.mean(diff))
    label   = "BAD" if score > EDGE_THRESHOLD else "GOOD"

    return crop, score, label


def make_panel(crop, score, label, view_name):
    """
    Build a 3-column panel: [raw crop | edge map | diff heatmap]
    with a header bar showing the view name, score, and verdict.
    """
    TARGET_H = 300

    # ── raw crop ──────────────────────────────────────────────────
    scale    = TARGET_H / crop.shape[0]
    tw       = max(1, int(crop.shape[1] * scale))
    raw_vis  = cv2.resize(crop, (tw, TARGET_H))

    # ── edge map ──────────────────────────────────────────────────
    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (REF_W, REF_H))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    edges   = cv2.Canny(blurred, 30, 100)
    edge_vis = cv2.cvtColor(cv2.resize(edges, (REF_W, TARGET_H)), cv2.COLOR_GRAY2BGR)

    # ── diff heatmap ──────────────────────────────────────────────
    edge_f   = edges.astype(np.float32) / 255.0
    diff     = np.abs(edge_f - ref_mean) / (ref_std + 0.1)
    diff_u8  = np.clip(diff * 255, 0, 255).astype(np.uint8)
    heat_vis = cv2.applyColorMap(cv2.resize(diff_u8, (REF_W, TARGET_H)), cv2.COLORMAP_HOT)

    # ── reference ─────────────────────────────────────────────────
    ref_u8   = (ref_mean * 255).astype(np.uint8)
    ref_vis  = cv2.cvtColor(cv2.resize(ref_u8, (REF_W, TARGET_H)), cv2.COLOR_GRAY2BGR)

    # ── column labels ─────────────────────────────────────────────
    def labelled(img, text):
        out = img.copy()
        cv2.putText(out, text, (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        return out

    row = np.hstack([
        labelled(raw_vis,  "crop"),
        labelled(edge_vis, "edges"),
        labelled(ref_vis,  "reference"),
        labelled(heat_vis, "diff"),
    ])

    # ── header bar ────────────────────────────────────────────────
    bar_color = (0, 60, 0) if label == "GOOD" else (0, 0, 80)
    txt_color = (0, 220, 0) if label == "GOOD" else (0, 80, 255)
    header    = np.full((32, row.shape[1], 3), bar_color, dtype=np.uint8)
    header_text = f"{view_name}   score={score:.3f}   threshold={EDGE_THRESHOLD}   → {label}"
    cv2.putText(header, header_text, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, txt_color, 1)

    return np.vstack([header, row])


# ── 1. Reference preview ──────────────────────────────────────────────────────

ref_u8  = (ref_mean * 255).astype(np.uint8)
ref_big = cv2.resize(ref_u8, (200, 600))
cv2.imshow("Reference edge map", ref_big)
cv2.waitKey(1)

# ── 2. Front and side views ───────────────────────────────────────────────────

print("\n── Front view ──")
front_path = input("Front image path: ").strip().strip('"')
front_crop, front_score, front_label = get_crop(front_path)

print("\n── Side view ──")
side_path = input("Side  image path: ").strip().strip('"')
side_crop, side_score, side_label = get_crop(side_path)

# ── 3. Build combined display ─────────────────────────────────────────────────

panels = []
if front_crop is not None:
    panels.append(make_panel(front_crop, front_score, front_label, "FRONT"))
    print(f"\n  Front: score={front_score:.3f}  → {front_label}")

if side_crop is not None:
    panels.append(make_panel(side_crop, side_score, side_label, "SIDE"))
    print(f"  Side:  score={side_score:.3f}  → {side_label}")

if panels:
    # stack front above side, separated by a thin divider
    divider = np.full((4, panels[0].shape[1], 3), 60, dtype=np.uint8)
    combined = panels[0]
    for p in panels[1:]:
        combined = np.vstack([combined, divider, p])

    # overall verdict banner
    all_labels = [l for l in [front_label, side_label] if l is not None]
    verdict    = "BAD" if "BAD" in all_labels else "GOOD"
    v_color    = (0, 0, 80) if verdict == "BAD" else (0, 60, 0)
    v_txt      = (0, 80, 255) if verdict == "BAD" else (0, 220, 0)
    banner     = np.full((40, combined.shape[1], 3), v_color, dtype=np.uint8)
    cv2.putText(banner, f"OVERALL: {verdict}", (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, v_txt, 2)
    combined   = np.vstack([combined, banner])

    cv2.imshow("Bottle debug — front + side", combined)
    cv2.waitKey(0)

cv2.destroyAllWindows()