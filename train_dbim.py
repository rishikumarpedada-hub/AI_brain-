import cv2
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_mmi(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    filtered = cv2.GaussianBlur(enhanced, (5,5), 0)
    return filtered

# -----------------------------
# Otsu Thresholding
# -----------------------------
def otsu_segmentation(image):
    _, thres = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thres

# -----------------------------
# Simplified DBIM Implementation
# -----------------------------
def forward_model(permittivity_map):
    # Approximate scattered fields using Fourier transform
    return np.fft.fft2(permittivity_map)

def dbim_segmentation(mmi_data, iterations=5, step_size=0.01, verbose=False):
    # Initialize permittivity map (uniform guess)
    permittivity_map = np.ones_like(mmi_data, dtype=np.float32)

    measured_fields = np.fft.fft2(mmi_data)  # treat preprocessed image as "measured"

    for it in range(iterations):
        simulated_fields = forward_model(permittivity_map)
        error = measured_fields - simulated_fields
        # Update permittivity map using error feedback
        permittivity_map += step_size * np.real(np.fft.ifft2(error))
        if verbose:
            print(f"DBIM Iter {it+1}, error norm: {np.linalg.norm(error)}")

    # Normalize and threshold to create mask
    permittivity_map = (permittivity_map - permittivity_map.min()) / (permittivity_map.max() - permittivity_map.min())
    mask = (permittivity_map > 0.5).astype(np.uint8)
    return mask

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(image):
    hist = cv2.calcHist([image],[0],None,[32],[0,256]).flatten()
    mean_val = np.mean(image)
    std_val = np.std(image)
    return np.concatenate([hist, [mean_val, std_val]])

# -----------------------------
# Load Dataset
# -----------------------------
def load_dataset(class_dirs):
    X, y = [], []
    label_to_name = {}
    name_to_label = {}

    for idx, (class_name, folder) in enumerate(class_dirs):
        name_to_label[class_name] = idx
        label_to_name[idx] = class_name
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            img = cv2.imread(path)
            if img is None:
                continue
            pre = preprocess_mmi(img)
            seg = dbim_segmentation(pre)
            features = extract_features(seg)
            X.append(features)
            y.append(idx)
    return np.array(X), np.array(y), label_to_name, name_to_label


def discover_class_dirs(base_dir="data"):
    preferred = ["non-stroke", "nonstroke", "ischemic", "hemorrhagic"]
    discovered = []

    for class_name in preferred:
        folder = os.path.join(base_dir, class_name)
        if os.path.isdir(folder):
            discovered.append((class_name, folder))

    # Backward-compatible fallback for old binary structure
    stroke_folder = os.path.join(base_dir, "stroke")
    nonstroke_folder = None
    for name in ("non-stroke", "nonstroke"):
        candidate = os.path.join(base_dir, name)
        if os.path.isdir(candidate):
            nonstroke_folder = candidate
            break
    if os.path.isdir(stroke_folder) and nonstroke_folder and len(discovered) < 2:
        discovered = [("non-stroke", nonstroke_folder), ("stroke", stroke_folder)]

    return discovered

# -----------------------------
# Train Model
# -----------------------------
if __name__ == "__main__":
    max_imbalance_ratio = 2.0
    auto_rebalance = True

    class_dirs = discover_class_dirs("data")
    if len(class_dirs) < 2:
        raise ValueError(
            "Need at least 2 class folders in data/. Use non-stroke+stroke or non-stroke+ischemic+hemorrhagic."
        )

    print("Training classes:")
    for class_name, folder in class_dirs:
        print(f"  {class_name}: {folder}")

    X, y, label_map, name_to_label = load_dataset(class_dirs)
    if len(X) < 10:
        raise ValueError("Dataset is too small. Add more images under data/<class_name>/ folders.")

    unique_labels, counts = np.unique(y, return_counts=True)
    if len(unique_labels) < 2 or np.min(counts) < 2:
        raise ValueError("Each class must have at least 2 samples for stratified training/evaluation.")

    print("Class Distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  {label_map.get(int(label), str(int(label)))}: {int(count)}")

    min_count = int(np.min(counts))
    max_count = int(np.max(counts))
    imbalance_ratio = max_count / max(min_count, 1)
    print(f"Imbalance Ratio (max/min): {imbalance_ratio:.2f}")
    if imbalance_ratio > max_imbalance_ratio:
        if not auto_rebalance:
            raise ValueError(
                "Class imbalance is too high (>2.0). Add more images to the minority class before training."
            )

        # Auto-rebalance by random undersampling majority classes to keep ratio <= max_imbalance_ratio.
        print("Imbalance too high. Applying automatic undersampling rebalance...")
        rng = np.random.default_rng(42)
        target_max_count = int(min_count * max_imbalance_ratio)
        keep_indices = []
        for label in unique_labels:
            label_idx = np.where(y == label)[0]
            if len(label_idx) > target_max_count:
                sampled = rng.choice(label_idx, size=target_max_count, replace=False)
                keep_indices.extend(sampled.tolist())
            else:
                keep_indices.extend(label_idx.tolist())
        keep_indices = np.array(sorted(keep_indices), dtype=int)
        X = X[keep_indices]
        y = y[keep_indices]

        unique_labels, counts = np.unique(y, return_counts=True)
        min_count = int(np.min(counts))
        max_count = int(np.max(counts))
        imbalance_ratio = max_count / max(min_count, 1)
        print("Class Distribution After Rebalance:")
        for label, count in zip(unique_labels, counts):
            print(f"  {label_map.get(int(label), str(int(label)))}: {int(count)}")
        print(f"Imbalance Ratio After Rebalance: {imbalance_ratio:.2f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    labels = sorted(label_map.keys())
    target_names = [label_map[i] for i in labels]
    print(classification_report(y_test, y_pred, labels=labels, target_names=target_names, zero_division=0))

    min_class_count = int(np.min(counts))
    cv_folds = min(5, min_class_count)
    if cv_folds >= 2:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
        print(f"\nCross-validation (f1_macro, {cv_folds}-fold): {cv_scores}")
        print(f"Mean CV f1_macro: {cv_scores.mean():.4f}")

    # Save model bundle with metadata for app/predict compatibility
    bundle = {
        "model": model,
        "label_to_name": label_map,
        "name_to_label": name_to_label,
        "feature_dim": int(X.shape[1]),
        "trained_at": str(np.datetime64("now")),
    }
    joblib.dump(bundle, "dbim_model.joblib")
    print("Model bundle saved as dbim_model.joblib")
