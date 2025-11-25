"""
Online IMU + image-based KNN prediction using streaming data collected by 2_GetData.py.

Important:
    This script must be executed at the same time as:
        → 2_GetData.py
    which captures IMU data and camera frames and writes Sample_* folders into Temporary_Data/.

    If 2_GetData.py is not running or no new samples are created, this script will keep waiting
    for data and will not terminate on its own.

Functionality:
    - Always runs in ONLINE mode (uses the most recent WINDOW_SIZE sample folders).
    - Extracts time- and frequency-domain features from dual-IMU data.
    - Extracts appearance-based image features (HOG, colour histograms, edge/contour metrics)
      from the captured frames when available.
    - Reindexes features to the column set defined in dataset_header_imu_img.csv.
    - Applies a pre-trained scaler and KNN classifier to produce action predictions.
    - Uses majority voting over the current window of samples for a more stable prediction.
    - Logs both the last stable "update" and the current prediction to an automatically
      incremented Excel file (Participant_N.xlsx) in Participant_Data/.
    - When a stable transition to a Grab or Down state is detected, prints a highlighted banner
      and pauses execution for 6 seconds before continuing.

Usage:
    - Ensure this script, the scaler and KNN .pkl files, the feature header CSV, and the
      Temporary_Data/ directory are located under the same project root.
    - Start 2_GetData.py first to generate Sample_* folders, then run this script.
    - Stop this script manually with Ctrl+C when you no longer need online predictions.

This script is provided as companion code for the associated project/publication.
"""


# if __name__ == "__main__":
#     print("\033cStarting...\n")
#     print("⚠️  Please ensure that '2_GetData.py' is running simultaneously.")
#     print("    This script will wait until new samples appear.\n")



import os
import time
import re
from collections import Counter
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from openpyxl import Workbook
from scipy.fft import fft
from scipy.stats import entropy, kurtosis, skew
from skimage.feature import hog


# ===========================
# CONFIGURATION
# ===========================

# Automatically set project root to the script's directory
PROJECT_ROOT = Path(__file__).resolve().parent

BASE_PATH = PROJECT_ROOT / "Temporary_Data"
PARTICIPANT_DIR = PROJECT_ROOT / "Participant_Data"

SCALER_PATH = PROJECT_ROOT / "scaler_imu_image.pkl"
KNN_MODEL_PATH = PROJECT_ROOT / "knn_model_imu_image.pkl"
FEATURE_HEADER_CSV = PROJECT_ROOT / "dataset_header_imu_img.csv"

LABEL_MAP = {0: "Grab", 1: "Walk", 2: "Down"}

WINDOW_SIZE = 3  # number of samples per decision


# ===========================
# Utility functions
# ===========================

def numerical_sort(value: str) -> int:
    """Sort strings by the first integer they contain (Sample_1 < Sample_10)."""
    numbers = re.findall(r"\d+", value)
    return int(numbers[0]) if numbers else float("inf")


def print_action_banner(action: str) -> None:
    """Print Grab/Walk/Down in bold and colour to highlight detection."""
    colours = {
        "Grab": "\033[93m",           # Yellow text
        "Down": "\033[95m",           # Magenta text
        "Walk": "\033[44m\033[97m",   # Blue background + white text
    }

    reset = "\033[0m"
    colour = colours.get(action, "")
    text = action.upper()

    if action in {"Grab", "Down"}:
        extra = "  (PAUSE 6 SECONDS)"
    else:
        extra = ""

    print(f"\n{colour}\033[1m=============== {text} DETECTED{extra} ==============={reset}\n")


def initialise_participant_workbook():
    """
    Ensure the Participant_Data directory exists, create a new participantN.xlsx
    (with headers) if needed, and return (excel_path, workbook, worksheet).
    """
    PARTICIPANT_DIR.mkdir(parents=True, exist_ok=True)

    num = 1
    excel_path = PARTICIPANT_DIR / f"participant{num}.xlsx"
    while excel_path.exists():
        num += 1
        excel_path = PARTICIPANT_DIR / f"participant{num}.xlsx"

    wb = Workbook()
    ws = wb.active
    ws.title = f"Participant{num}"
    ws.append(["Update", "Current"])

    print(f"Using participant file: {excel_path}")
    return excel_path, wb, ws


def extract_imu_features(df: pd.DataFrame) -> pd.Series:
    """Compute statistical and spectral features from IMU dataframe."""
    feats = {}

    for col in df.columns:
        sig = df[col].values
        feats[f"{col}_mean"] = np.mean(sig)
        feats[f"{col}_std"] = np.std(sig)
        feats[f"{col}_min"] = np.min(sig)
        feats[f"{col}_max"] = np.max(sig)
        feats[f"{col}_range"] = np.ptp(sig)
        feats[f"{col}_median"] = np.median(sig)
        feats[f"{col}_skew"] = skew(sig)
        feats[f"{col}_kurtosis"] = kurtosis(sig)
        feats[f"{col}_energy"] = np.sum(sig ** 2)
        feats[f"{col}_rms"] = np.sqrt(np.mean(sig ** 2))

        hist, _ = np.histogram(sig, bins=20, density=True)
        feats[f"{col}_entropy"] = entropy(hist + 1e-6)

        fft_vals = np.abs(fft(sig))[: len(sig) // 2]
        top_fft = np.sort(fft_vals)[-3:]
        for i, peak in enumerate(top_fft):
            feats[f"{col}_fft_peak_{i + 1}"] = peak

    return pd.Series(feats)


def extract_image_features(image_paths: list[str]) -> pd.Series:
    """Extract HOG + colour histogram + edge/contour features from images."""
    features = []

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (128, 128))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        hist = cv2.calcHist(
            [img], [0, 1, 2], None,
            [8, 8, 8],
            [0, 256, 0, 256, 0, 256],
        )
        hist = cv2.normalize(hist, hist).flatten()

        hog_vec = hog(
            gray,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=False,
        )

        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges > 0)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area, perimeter = 0.0, 0.0
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(largest))
            perimeter = float(cv2.arcLength(largest, True))

        features.append(
            np.concatenate([hist, hog_vec, [edge_density, area, perimeter]])
        )

    if not features:
        return pd.Series(dtype="float64")

    mean = np.mean(np.array(features), axis=0)
    return pd.Series(mean, index=[f"imgfeat_{i}" for i in range(len(mean))])


# ===========================
# Main prediction loop (ONLINE ONLY)
# ===========================

def main() -> None:
    # Load models and feature configuration
    print("Loading scaler and KNN model...")
    scaler = joblib.load(SCALER_PATH)
    knn = joblib.load(KNN_MODEL_PATH)

    reference_columns = (
        pd.read_csv(FEATURE_HEADER_CSV)
        .drop(columns=["label"], errors="ignore")
        .columns
    )

    # Prepare participant Excel workbook
    excel_path, wb, ws = initialise_participant_workbook()

    # Mode is fixed to Online
    print("Mode selected: Online\n")
    print("🟢 Real-time Prediction with Averaging Started...")

    current_vote = "Down"
    motor_state = "Down"  # purely logical now
    state_assurance: list[str] = []

    try:
        while True:
            start_tracking_time = time.time()

            if not BASE_PATH.exists():
                print(f"Base path '{BASE_PATH}' does not exist. Waiting...")
                time.sleep(1.0)
                continue

            folders = sorted(
                [f for f in os.listdir(BASE_PATH) if (BASE_PATH / f).is_dir()],
                key=numerical_sort,
            )

            if not folders:
                print("No folders found in Temporary_Data. Waiting for new samples...")
                time.sleep(1.0)
                continue

            # ONLINE MODE: always use the latest WINDOW_SIZE folders
            if len(folders) <= WINDOW_SIZE:
                recent_folders = folders
            else:
                recent_folders = folders[-WINDOW_SIZE:]

            predictions: list[int] = []
            samples_used: list[str] = []

            for folder in recent_folders:
                folder_path = BASE_PATH / folder
                imu_file = folder_path / "imu.csv"
                image_files = [
                    str(folder_path / f)
                    for f in os.listdir(folder_path)
                    if f.lower().endswith(".jpg")
                ]

                if not imu_file.exists() or imu_file.stat().st_size == 0:
                    continue

                try:
                    imu_df = pd.read_csv(imu_file, header=None)
                    sensor_columns = [
                        "acc_x_1", "acc_y_1", "acc_z_1",
                        "gyro_x_1", "gyro_y_1", "gyro_z_1",
                        "acc_x_2", "acc_y_2", "acc_z_2",
                        "gyro_x_2", "gyro_y_2", "gyro_z_2",
                    ]
                    imu_df.columns = sensor_columns
                except Exception as e:
                    print(f"⚠️ Skipping folder '{folder}': {e}")
                    continue

                imu_feats = extract_imu_features(imu_df.iloc[:10])
                img_feats = extract_image_features(image_files)
                combined_feats = pd.concat([imu_feats, img_feats])

                input_df = (
                    pd.DataFrame([combined_feats])
                    .reindex(columns=reference_columns, fill_value=0)
                )

                scaled = scaler.transform(input_df)
                pred = int(knn.predict(scaled)[0])
                predictions.append(pred)
                samples_used.append(folder)

            elapsed = time.time() - start_tracking_time
            print(f"\nProcessing time this cycle: {elapsed:.3f} s")

            if not predictions:
                print("⚠️ No valid samples found in latest folders.")
                time.sleep(0.5)
                continue

            vote = Counter(predictions).most_common(1)[0][0]
            predicted_label = LABEL_MAP[vote]
            print(f"Prediction: {predicted_label}")
            print(f"📂 Samples used: {', '.join(samples_used)}")

            state_assurance.append(predicted_label)
            update = ""

            if len(state_assurance) > 2:
                last_two = state_assurance[-2:]
                print(f"State assurance (last two): {last_two}")

                if last_two[0] == last_two[1]:
                    next_vote = predicted_label
                    if next_vote != current_vote:
                        if current_vote in {"Grab", "Down"}:
                            motor_state = current_vote

                        if motor_state != next_vote or next_vote == "Walk":
                            # Stable updated prediction
                            print(f"\n✨ Updated prediction (stable): {next_vote}")
                            update = next_vote
                            print(f"📂 Samples used for update: {', '.join(samples_used)}")

                            # Highlight Grab/Down and pause 6 seconds
                            if next_vote in {"Grab", "Down"}:
                                print_action_banner(next_vote)
                                time.sleep(6)

                            print(f"{next_vote} complete.\n")

                        current_vote = next_vote

            # Log to Excel
            ws.append([update, predicted_label])

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Stopping...")

    wb.save(excel_path)
    print(f"Excel file saved successfully at: {excel_path}")


if __name__ == "__main__":
    print("\033cStarting...\n")
    print("      ⚠️ Please ensure that '2_GetData.py' is running simultaneously.")
    print("    This script will read only last three samples until new ones appear.\n")
    time.sleep(5)
    main()
