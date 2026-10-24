import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from pathlib import Path
import os


def calculate_and_plot(mode, output_dir):
    try:
        df = pd.read_csv(f"evaluation/results/raw_predictions_{mode}.csv")
        gt = pd.read_csv("data/raw/labels.csv")

        merged = pd.merge(df, gt, on="image_path", how="inner")
        if merged.empty:
            return

        y_true = merged["is_drowsy"].values
        y_pred = (merged["predicted_state"] == "DROWSY").astype(int).values

        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["NON_DROWSY", "DROWSY"],
            yticklabels=["NON_DROWSY", "DROWSY"],
        )
        plt.title(f"Ma Trận Nhầm Lẫn - {mode.upper()}")
        plt.ylabel("Thực tế")
        plt.xlabel("Dự đoán")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix_{mode}.png")
        plt.close()

        # Tạo score giả lập cho đường cong ROC/PR vì mô hình trả về nhãn cứng
        y_score = y_pred.astype(float)

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Tỷ lệ Dương tính Giả (False Positive Rate)")
        plt.ylabel("Tỷ lệ Dương tính Thật (True Positive Rate)")
        plt.title(f"Biểu Đồ ROC - {mode.upper()}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/roc_curve_{mode}.png")
        plt.close()

        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        plt.figure(figsize=(6, 5))
        plt.plot(
            recall,
            precision,
            color="blue",
            lw=2,
            label=f"PR curve (AUC = {pr_auc:.2f})",
        )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall (Độ nhạy)")
        plt.ylabel("Precision (Độ chính xác)")
        plt.title(f"Biểu Đồ Precision-Recall - {mode.upper()}")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pr_curve_{mode}.png")
        plt.close()

    except FileNotFoundError:
        pass


output_dir = "evaluation/metrics"
os.makedirs(output_dir, exist_ok=True)
calculate_and_plot("mode_a", output_dir)
calculate_and_plot("mode_b", output_dir)
