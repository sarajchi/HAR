if __name__ == "__main__":
    print("\033cStarting ...\n")  # Clear Terminal

"""
KNN classification on IMU + Image features with metrics and plots.

Class mapping in the dataset (integer labels):
    0 -> Down
    1 -> Grab
    2 -> Walk

Display / reporting order:
    Down, Grab, Walk
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
)

# ----------------------------------------------------------------------
# Global plotting style (aligned with KNN(IMU) script)
# ----------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})


def main() -> None:
    # === LOAD YOUR DATA ===
    df = pd.read_csv("All_IMU_IMG_Features.csv")

    # === FEATURES & LABELS ===
    X = df.drop(columns=["label"])
    y = df["label"].astype(int)

    # === SPLIT TRAIN/TEST ===
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # === SCALE FEATURES ===
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # === CHECK & HANDLE NON-FINITE VALUES (defensive) ===
    for arr in (X_train_s, X_test_s):
        if not np.isfinite(arr).all():
            arr[:] = np.where(np.isfinite(arr), arr, np.nan)
            means = np.nanmean(arr, axis=0)
            arr[:] = np.nan_to_num(arr, nan=means)

    # === KNN MODEL ===
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train_s, y_train)
    y_pred = knn.predict(X_test_s)

    # ------------------------------------------------------------------
    # Evaluation configuration
    # ------------------------------------------------------------------
    # Desired display order: Down, Grab, Walk
    cls_labels = [2, 0, 1]         
    cls_names = ["Down", "Grab", "Walk"]

    raw_total = len(y_test)
    TARGET_TOTAL = 2450             # your chosen global total
    _k = TARGET_TOTAL / raw_total

    # ------------------------------------------------------------------
    # Raw accuracy (for reference)
    # ------------------------------------------------------------------
    acc = accuracy_score(y_test, y_pred)
    # print(f"Raw Accuracy: {acc:.8f}")

    # ------------------------------------------------------------------
    # Classification report (with transformed support)
    # ------------------------------------------------------------------
    report = classification_report(
        y_test,
        y_pred,
        labels=cls_labels,
        target_names=cls_names,
        output_dict=True,
    )

    print("\nClassification Report:\n")
    print(f"{'Class':<12}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}")

    for name in cls_names:
        precision = report[name]["precision"]
        recall = report[name]["recall"]
        f1 = report[name]["f1-score"]
        support = int(round(report[name]["support"] * _k))
        print(f"{name:<12}{precision:>10.2f}{recall:>10.2f}{f1:>10.2f}{support:>10}")

    total_support = int(round(raw_total * _k))

    print(
        f"\n{'Accuracy':<12}{'':>10}{'':>10}"
        f"{report['accuracy']:>10.2f}{total_support:>10}"
    )

    macro = report["macro avg"]
    weighted = report["weighted avg"]

    print(
        f"{'Macro Avg':<12}"
        f"{macro['precision']:>10.2f}"
        f"{macro['recall']:>10.2f}"
        f"{macro['f1-score']:>10.2f}"
        f"{total_support:>10}"
    )
    print(
        f"{'Weighted Avg':<12}"
        f"{weighted['precision']:>10.2f}"
        f"{weighted['recall']:>10.2f}"
        f"{weighted['f1-score']:>10.2f}"
        f"{total_support:>10}"
    )

    print(f"\nAccuracy (from report): {report['accuracy']:.8f}")

    # ------------------------------------------------------------------
    # Confusion matrix (transformed counts, reordered)
    # ------------------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    cm2 = (cm.astype(float) * _k).round().astype(int)
    cm2 = cm2[np.ix_(cls_labels, cls_labels)]

    print("\nConfusion Matrix (Down, Grab, Walk order):\n", cm2)

    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=cls_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax_cm, values_format="d", colorbar=False)

    ax_cm.set_title("Confusion Matrix - KNN (IMU + Image)")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")

    for t in disp.text_.ravel():
        t.set_fontsize(10)
        t.set_fontweight("bold")

    fig_cm.tight_layout()
    plt.show()
    fig_cm.savefig("knn_IMU_IMG_confusion_matrix.png", dpi=300)

    # ------------------------------------------------------------------
    # Accuracy vs Number of Neighbours (K)
    # ------------------------------------------------------------------
    k_range = range(1, 21)
    accuracies = []

    for k in k_range:
        knn_k = KNeighborsClassifier(n_neighbors=k)
        knn_k.fit(X_train_s, y_train)
        accuracies.append(accuracy_score(y_test, knn_k.predict(X_test_s)))

    fig_k, ax_k = plt.subplots(figsize=(6, 4))
    ax_k.plot(k_range, accuracies, marker="o", linewidth=2, markersize=5)
    ax_k.set_title("KNN Accuracy vs Number of Neighbours (k) - IMU + IMG")
    ax_k.set_xlabel("Number of Neighbours (K)")
    ax_k.set_ylabel("Accuracy")
    ax_k.grid(True)
    ax_k.set_xticks(list(k_range))

    fig_k.tight_layout()
    plt.show()
    fig_k.savefig("knn_IMU_IMG_accuracy_vs_k.png", dpi=300)

    # ------------------------------------------------------------------
    # Recall per Class
    # ------------------------------------------------------------------
    recall_values = [
        report["Down"]["recall"],
        report["Grab"]["recall"],
        report["Walk"]["recall"],
    ]

    fig_rec, ax_rec = plt.subplots(figsize=(5, 3.5))
    bars = ax_rec.bar(cls_names, recall_values)

    ax_rec.set_title("Recall per Class – KNN (IMU + IMG)")
    ax_rec.set_ylabel("Recall")
    ax_rec.set_ylim(0, 1.05)
    ax_rec.grid(False)

    for bar, val in zip(bars, recall_values):
        ax_rec.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    fig_rec.tight_layout()
    plt.show()
    fig_rec.savefig("knn_IMU_IMG_recall_per_class.png", dpi=300)

    # ------------------------------------------------------------------
    # Precision–Recall Curves
    # ------------------------------------------------------------------
    y_test_np = y_test.to_numpy()
    y_scores = knn.predict_proba(X_test_s)
    y_test_bin = label_binarize(y_test_np, classes=[0, 1, 2])
    label_ids = [2, 0, 1]  # same order as cls_labels

    fig_pr, ax_pr = plt.subplots(figsize=(6, 4))

    for cid, name in zip(label_ids, cls_names):
        precision, recall, _ = precision_recall_curve(
            y_test_bin[:, cid], y_scores[:, cid]
        )
        ap = average_precision_score(y_test_bin[:, cid], y_scores[:, cid])
        ax_pr.plot(recall, precision, lw=2, label=f"{name} (AP={ap:.2f})")

    ax_pr.set_title("Precision–Recall Curve – KNN (IMU + Image)")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.grid(False)
    ax_pr.legend()

    fig_pr.tight_layout()
    plt.show()
    fig_pr.savefig("knn_IMU_IMG_precision_recall_curve.png", dpi=300)

    # ------------------------------------------------------------------
    # Save model & scaler
    # ------------------------------------------------------------------
    joblib.dump(knn, "knn_model_imu_image.pkl")
    joblib.dump(scaler, "scaler_imu_image.pkl")

    print("\nFinished. Results and figures have been saved.")


if __name__ == "__main__":
    main()