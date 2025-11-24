if __name__ == "__main__":
    print("\033cStarting ...\n")  # Clear Terminal


import os
import cv2
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.fft import fft
from skimage.feature import hog
import warnings

# === Suppress warnings like precision loss ===
warnings.filterwarnings("ignore", category=RuntimeWarning)

# === CONFIGURATION ===
BASE_PATH = r"C:\Users\m-sarajchi\OneDrive - UWE Bristol\PCUWE\Upper Limb Exoskeleton\Dataset_Evan_Updated\Dataset"
BASE_PATH = r"C:\Users\m-sarajchi\OneDrive - UWE Bristol\PCUWE\Upper Limb Exoskeleton\Dataset\Dataset"
LABELS = {"grab": 0, "walk": 1, "down": 2}
WINDOW_SIZE = 10
STEP_SIZE = 5   ###overlap = 50%
# === FEATURE FUNCTIONS ===
def extract_full_imu_features(row_df: pd.DataFrame) -> pd.Series:
    feats = {}
    row_df = row_df.drop(columns=["time_stamp"], errors="ignore")
    for col in row_df.columns:
        sig = row_df[col].values
        feats[f"{col}_mean"] = np.mean(sig)
        feats[f"{col}_std"] = np.std(sig)
        feats[f"{col}_min"] = np.min(sig)
        feats[f"{col}_max"] = np.max(sig)
        feats[f"{col}_range"] = np.ptp(sig)
        feats[f"{col}_median"] = np.median(sig)
        feats[f"{col}_skew"] = np.nan_to_num(skew(sig), nan=0.0)
        feats[f"{col}_kurtosis"] = np.nan_to_num(kurtosis(sig, bias=False), nan=0.0)
        feats[f"{col}_energy"] = np.sum(sig ** 2)
        feats[f"{col}_rms"] = np.sqrt(np.mean(sig ** 2))
        hist, _ = np.histogram(sig, bins=20, density=True)
        feats[f"{col}_entropy"] = entropy(hist + 1e-6)
        fft_vals = np.abs(fft(sig))[: len(sig) // 2]
        top = np.sort(fft_vals)[-3:][::-1]
        while len(top) < 3:
            top = np.append(top, np.nan)
        for i, pk in enumerate(top, 1):
            feats[f"{col}_fft_peak_{i}"] = pk
    return pd.Series(feats)

def extract_image_features_from_paths(image_paths: list) -> pd.Series:
    all_features = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (128, 128))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hog_vec = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                      cells_per_block=(2, 2), block_norm="L2-Hys",
                      visualize=False, feature_vector=True)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges > 0)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = perimeter = 0.0
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)
        all_features.append(np.concatenate([hist, hog_vec, [edge_density, area, perimeter]]))
    if not all_features:
        return pd.Series(dtype='float64')
    mean_features = np.mean(np.array(all_features), axis=0)
    feature_names = [f"imgfeat_{i}" for i in range(len(mean_features))]
    return pd.Series(mean_features, index=feature_names)

# === PROCESS ALL HIGH-LEVEL FOLDERS ===
rows = []

for activity, label in LABELS.items():
    activity_path = os.path.join(BASE_PATH, activity)
    imu_all = []
    image_all = []

    folder_names = sorted([f for f in os.listdir(activity_path) if os.path.isdir(os.path.join(activity_path, f))])
    for folder in folder_names:
        subfolder_path = os.path.join(activity_path, folder)
        imu_path = os.path.join(subfolder_path, "imu.csv")
        if not os.path.exists(imu_path):
            continue
        imu_df = pd.read_csv(imu_path)
        image_files = sorted([f for f in os.listdir(subfolder_path) if f.lower().endswith(".jpg")])
        image_paths = [os.path.join(subfolder_path, f) for f in image_files]
        imu_all.extend(imu_df.to_dict("records"))
        image_all.extend(image_paths)

    imu_full_df = pd.DataFrame(imu_all)
    imu_columns = [col for col in imu_full_df.columns if col != "time_stamp"]
    n_windows = (min(len(imu_full_df), len(image_all)) - WINDOW_SIZE) // STEP_SIZE + 1

    for i in range(n_windows):
        s = i * STEP_SIZE
        e = s + WINDOW_SIZE
        imu_window = imu_full_df.iloc[s:e]
        img_window = image_all[s:e]
        imu_feat = extract_full_imu_features(imu_window)
        img_feat = extract_image_features_from_paths(img_window)
        imu_raw = imu_full_df.iloc[s][imu_columns].to_dict()
        imu_raw = {f"imu_raw_{k}": v for k, v in imu_raw.items()}
        row = pd.concat([imu_feat, img_feat])
        row["image_names"] = ";".join([os.path.basename(p) for p in img_window])
        row["imu_window_start"] = s
        row["imu_window_end"] = e - 1
        row["label"] = label
        row = pd.concat([row, pd.Series(imu_raw)])
        rows.append(row)

# === SAVE OUTPUT ===
out = pd.DataFrame(rows)
out.to_csv("imu_image_features_all_combined_Updated_Haocheng.csv", index=False)
print("✅ Saved to imu_image_features_all_combined_Updated_Haocheng.csv")
