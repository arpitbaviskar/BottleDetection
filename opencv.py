import cv2
import numpy as np
import os

GOOD_DIR = r"E:\Dalnex\sample_dataset\sample_dataset\Good_images"
BAD_DIR  = r"E:\Dalnex\sample_dataset\sample_dataset\Bad_images"

# STEP 1: Describe each image using 3 numbers
def get_features(img):
    img  = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    texture = cv2.Laplacian(gray, cv2.CV_64F).var()   # how complex the image is
    edges   = cv2.Canny(gray, 50, 150).mean()          # how many edges

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shape = 1.0
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        shape = w / h   # tall bottle → small, wide bottle → big

    return [texture, edges, shape]


# STEP 2: Train KNN on all 40 images
def train():
    X, y = [], []

    for file in os.listdir(GOOD_DIR):
        img = cv2.imread(os.path.join(GOOD_DIR, file))
        if img is not None:
            X.append(get_features(img))
            y.append(0)              # 0 = GOOD

    for file in os.listdir(BAD_DIR):
        img = cv2.imread(os.path.join(BAD_DIR, file))
        if img is not None:
            X.append(get_features(img))
            y.append(1)              # 1 = BAD

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Normalise so all features are on the same scale
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8
    X    = (X - mean) / std

    knn = cv2.ml.KNearest_create()
    knn.train(X, cv2.ml.ROW_SAMPLE, y.reshape(-1, 1))

    return knn, mean, std


# STEP 3: Predict GOOD or BAD for a new image
def predict(img, knn, mean, std):
    features = np.array([get_features(img)], dtype=np.float32)
    features = (features - mean) / std

    _, result, _, _ = knn.findNearest(features, k=1)

    if int(result[0][0]) == 0:
        return "GOOD", (34, 197, 94)   # green
    else:
        return "BAD",  (0, 0, 220)     # red


# STEP 4: Find a tight bounding box around the bottle
# Strategy: try multiple brightness thresholds and pick the one that gives
# the most "bottle-like" box — not full-image-width, not full-image-height
def find_bottle(img):
    h, w  = img.shape[:2]
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    best_box   = None
    best_score = 0

    for thresh_val in range(140, 220, 10):
        # Threshold: pixels brighter than thresh_val = white bottle
        _, mask = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)

        # Clean up the mask — fill holes, remove small blobs
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Skip tiny contours (less than 4% of image area)
        valid = [c for c in contours if cv2.contourArea(c) > h * w * 0.04]
        if not valid:
            continue

        largest      = max(valid, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest)

        # Score: reward large area, penalise boxes that span the full image width/height
        # (a full-width box means we caught the background, not the bottle)
        score = cv2.contourArea(largest) * (1 - bw/w) * (1 - bh/h * 0.3)

        if score > best_score:
            best_score = score
            best_box   = (x, y, bw, bh)

    # Fallback: whole image if nothing found
    if best_box is None:
        return 10, 10, w - 20, h - 20

    return best_box


# STEP 5: Draw the result and open a popup window
def show_result(img, label, bbox, color, filename):
    x, y, w, h   = bbox
    img_h, img_w = img.shape[:2]

    # Draw bounding box tightly around the bottle
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)

    # Draw filled label badge just above the box
    badge_y = max(0, y - 40)
    cv2.rectangle(img, (x, badge_y), (x + w, y), color, -1)
    cv2.putText(img, label, (x + 8, badge_y + 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw status bar at the bottom
    cv2.rectangle(img, (0, img_h - 36), (img_w, img_h), (0, 0, 0), -1)
    cv2.putText(img, f"File: {filename}  |  Result: {label}",
                (10, img_h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Open popup window
    cv2.imshow(f"{filename}  ->  {label}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# MAIN
if __name__ == "__main__":

    print("Training...")
    knn, mean, std = train()
    print("Done!")

    path = input("Enter image path: ").strip()
    img  = cv2.imread(path)

    if img is None:
        print("Cannot read image.")
        exit()

    # Resize if too large for screen
    h, w  = img.shape[:2]
    scale = min(900 / w, 900 / h, 1.0)
    img   = cv2.resize(img, (int(w * scale), int(h * scale)))

    label, color = predict(img, knn, mean, std)
    bbox         = find_bottle(img)
    filename     = os.path.basename(path)

    print(f"Result: {label}")
    show_result(img, label, bbox, color, filename)
