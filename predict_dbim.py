import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt

def preprocess_mmi(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    filtered = cv2.GaussianBlur(enhanced, (5,5), 0)
    return filtered

def forward_model(permittivity_map):
    return np.fft.fft2(permittivity_map)

def dbim_segmentation(mmi_data, iterations=5, step_size=0.01):
    permittivity_map = np.ones_like(mmi_data, dtype=np.float32)
    measured_fields = np.fft.fft2(mmi_data)

    for it in range(iterations):
        simulated_fields = forward_model(permittivity_map)
        error = measured_fields - simulated_fields
        permittivity_map += step_size * np.real(np.fft.ifft2(error))

    permittivity_map = (permittivity_map - permittivity_map.min()) / (permittivity_map.max() - permittivity_map.min())
    mask = (permittivity_map > 0.5).astype(np.uint8)
    return mask

def extract_features(image):
    hist = cv2.calcHist([image],[0],None,[32],[0,256]).flatten()
    mean_val = np.mean(image)
    std_val = np.std(image)
    return np.concatenate([hist, [mean_val, std_val]])

def visualize_segmentation(original, mask, label):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.imshow(mask, cmap='jet', alpha=0.5)
    plt.title(f"Prediction: {label}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    loaded = joblib.load("dbim_model.joblib")
    if isinstance(loaded, dict) and "model" in loaded:
        model = loaded["model"]
        label_to_name = loaded.get("label_to_name", {})
    else:
        model = loaded
        label_to_name = {0: "non-stroke", 1: "stroke"}

    img = cv2.imread("test1.png")  # replace with your test image
    pre = preprocess_mmi(img)
    seg = dbim_segmentation(pre)
    features = extract_features(seg).reshape(1, -1)

    prediction = model.predict(features)[0]
    label = label_to_name.get(int(prediction), str(prediction)).replace("_", " ").title()
    lesion_ratio = float(np.mean(seg))
    lesion_pixels = int(np.sum(seg))
    total_pixels = int(seg.size)
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(features)[0]))

    print(f"Prediction: {label}")
    if confidence is not None:
        print(f"Confidence: {confidence * 100:.2f}%")
    print(f"Lesion burden (mask): {lesion_ratio * 100:.2f}% ({lesion_pixels}/{total_pixels} pixels)")

    visualize_segmentation(img, seg, label)
