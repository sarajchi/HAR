if __name__ == "__main__":
    print("\033cStarting IMU feature extraction...\n")  # Clear terminal (most terminals)

"""
IMU feature extraction script.

This script:
    1. Loads segmented IMU data from a CSV file.
    2. Extracts a set of statistical and spectral features for each segment.
    3. Saves the resulting feature matrix to disk as `All_IMU_Features.csv`.

Assumptions:
    - The input CSV file `Segmented_Data.csv` contains stacked segments.
    - Each segment consists of a fixed number of consecutive rows (SEGMENT_SIZE).
    - The first two columns are non-signal metadata (e.g. index, timestamp).
    - The last column is the class label (`label`).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Not used, but kept if you later extend with plots

from pathlib import Path
from typing import List

from scipy.stats import skew, kurtosis, entropy
from scipy.fft import fft


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
INPUT_FILE = Path("Segmented_Data.csv")
OUTPUT_FILE = Path("All_IMU_Features.csv")

SEGMENT_SIZE = 10          # Number of samples per segment
ENTROPY_BINS = 20          # Histogram bins for entropy calculation
EPSILON = 1e-6             # Small value to avoid log(0) in entropy
N_FFT_PEAKS = 3            # Number of largest FFT peaks to store


# ----------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------
def extract_features(segment: pd.DataFrame) -> pd.Series:
    """
    Extracts time–domain and frequency–domain features from a single IMU segment.

    Parameters
    ----------
    segment : pd.DataFrame
        Segment of IMU data. Assumes:
            - First two columns are metadata (e.g. index, timestamp).
            - Last column is the label.
            - All intermediate columns are sensor channels.

    Returns
    -------
    pd.Series
        One row of features with the corresponding label.
    """
    features = {}

    # Exclude the first two metadata columns and the last label column
    feature_columns: List[str] = list(segment.columns[2:-1])

    for col in feature_columns:
        signal = segment[col].to_numpy()

        # Basic statistics
        features[f"{col}_mean"] = float(np.mean(signal))
        features[f"{col}_std"] = float(np.std(signal))
        features[f"{col}_min"] = float(np.min(signal))
        features[f"{col}_max"] = float(np.max(signal))
        features[f"{col}_range"] = float(np.max(signal) - np.min(signal))
        features[f"{col}_median"] = float(np.median(signal))

        # Higher-order statistics
        features[f"{col}_skew"] = float(skew(signal, bias=False, nan_policy="omit"))
        features[f"{col}_kurtosis"] = float(kurtosis(signal, bias=False, nan_policy="omit"))

        # Energy and RMS
        features[f"{col}_energy"] = float(np.sum(signal ** 2))
        features[f"{col}_rms"] = float(np.sqrt(np.mean(signal ** 2)))

        # Entropy (histogram-based)
        hist, _ = np.histogram(signal, bins=ENTROPY_BINS, density=True)
        features[f"{col}_entropy"] = float(entropy(hist + EPSILON))

        # FFT top peaks (magnitude spectrum, first half)
        fft_vals = np.abs(fft(signal))[: len(signal) // 2]
        if fft_vals.size > 0:
            top_fft = np.sort(fft_vals)[-N_FFT_PEAKS:]
        else:
            top_fft = np.zeros(N_FFT_PEAKS, dtype=float)

        for i, peak in enumerate(top_fft, start=1):
            features[f"{col}_fft_peak_{i}"] = float(peak)

    # Preserve the label for this segment (assumes constant label within segment)
    features["label"] = segment["label"].iloc[0]

    return pd.Series(features)


# ----------------------------------------------------------------------
# Main processing pipeline
# ----------------------------------------------------------------------
def main() -> None:
    # --- Load segmented data ---
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE.resolve()}")

    segmented_df = pd.read_csv(INPUT_FILE)

    if segmented_df.empty:
        raise ValueError(f"Input file {INPUT_FILE} is empty. Nothing to process.")

    # --- Extract features for each segment ---
    features_list = []

    # Iterate over stacked segments
    for start_idx in range(0, len(segmented_df), SEGMENT_SIZE):
        segment = segmented_df.iloc[start_idx : start_idx + SEGMENT_SIZE]

        # Skip incomplete trailing segments (optional – keep or remove as needed)
        if len(segment) < SEGMENT_SIZE:
            continue

        features = extract_features(segment)
        features_list.append(features)

    if not features_list:
        raise RuntimeError(
            "No segments were processed. Check SEGMENT_SIZE and input data format."
        )

    # --- Create final dataset with features ---
    features_df = pd.DataFrame(features_list)
    features_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Feature extraction completed successfully.")
    print(f"Number of segments processed: {len(features_df)}")
    print(f"Output saved to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()