"""Sinh báo cáo benchmark (Markdown) từ các file CSV trong evaluation/results.

Cấu trúc thư mục mong đợi:
    evaluation/
    ├── results/   <- comparison_report.csv, raw_predictions_*.csv, robustness_results.csv
    └── reports/   <- generate_benchmark.py (file này) và benchmark_report.md (đầu ra)

Chạy:  python generate_benchmark.py
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------- Cấu hình
REPORTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = REPORTS_DIR.parent / "results"
OUTPUT_FILE = REPORTS_DIR / "benchmark_report.md"

# Nhãn thật được suy ra từ tên thư mục chứa ảnh: .../<Subject>/<Class>/<id>.jpg
# DROWSY là lớp dương (positive). Điều chỉnh nếu quy ước nhãn của nhóm khác.
CLASS_TO_LABEL = {
    "Eyeclose": "DROWSY",
    "Yawn": "DROWSY",
    "Neutral": "NON_DROWSY",
    "Happy": "NON_DROWSY",
}
POSITIVE = "DROWSY"

MODES = {
    "eye_only": ("Eye only (EAR)", "raw_predictions_eye_only.csv"),
    "eye_yawn": ("Eye + Yawn (EAR + MAR)", "raw_predictions_eye_yawn.csv"),
}
LAT_COLS = {
    "Face Detection": "latency_face_ms",
    "Landmark / Feature Extraction (EAR, MAR)": "latency_landmark_ms",
    "Decision Engine": "latency_decision_ms",
    "**Tổng pipeline**": "latency_total_ms",
}


# ---------------------------------------------------------------- Tiện ích
def read_csv(name: str):
    path = RESULTS_DIR / name
    if not path.exists():
        print(f"[!] Không tìm thấy {path}")
        return None
    return pd.read_csv(path)


def md_table(df: pd.DataFrame) -> str:
    """DataFrame -> bảng Markdown (không cần thư viện tabulate)."""
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(map(str, cols)) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def load_predictions(file_name: str):
    df = read_csv(file_name)
    if df is None:
        return None
    parts = df["image_path"].str.replace("\\", "/", regex=False).str.split("/")
    df["cls"] = parts.str[-2]
    df["subject"] = parts.str[-3]
    df["y_true"] = df["cls"].map(CLASS_TO_LABEL)
    unknown = df.loc[df["y_true"].isna(), "cls"].unique()
    if len(unknown):
        print(f"[!] Bỏ qua lớp chưa có trong CLASS_TO_LABEL: {list(unknown)}")
        df = df.dropna(subset=["y_true"])
    return df


def metrics(df: pd.DataFrame) -> dict:
    t = df["y_true"] == POSITIVE
    p = df["predicted_state"] == POSITIVE
    tp, fp = int((t & p).sum()), int((~t & p).sum())
    fn, tn = int((t & ~p).sum()), int((~t & ~p).sum())
    div = lambda a, b: a / b if b else 0.0
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": len(df),
        "acc": div(tp + tn, len(df)),
        "prec": prec,
        "rec": rec,
        "f1": div(2 * prec * rec, prec + rec),
        "spec": div(tn, tn + fp),
    }


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


# ---------------------------------------------------------------- Các phần báo cáo
def section_overview(preds: dict) -> str:
    any_df = next(iter(preds.values()))
    dist = any_df["cls"].value_counts()
    rows = pd.DataFrame(
        {
            "Lớp": dist.index,
            "Nhãn": [CLASS_TO_LABEL[c] for c in dist.index],
            "Số ảnh": dist.values,
        }
    )
    subj = ", ".join(
        f"{s} ({n})" for s, n in any_df["subject"].value_counts().sort_index().items()
    )
    det = any_df["face_detected"].mean()
    return f"""## 1. Tổng quan tập đánh giá
- **Tổng số ảnh**: {len(any_df)}
- **Số đối tượng (subject)**: {any_df["subject"].nunique()} — {subj}
- **Tỉ lệ phát hiện được khuôn mặt**: {pct(det)}
- **Lớp dương (positive)**: `{POSITIVE}`

{md_table(rows)}
"""


def section_mode_comparison(preds: dict) -> str:
    rows, m = [], {}
    for key, (label, _) in MODES.items():
        if key not in preds:
            continue
        m[key] = metrics(preds[key])
        x = m[key]
        rows.append(
            {
                "Chế độ": label,
                "Accuracy": f"{x['acc']:.4f}",
                "Precision": f"{x['prec']:.4f}",
                "Recall": f"{x['rec']:.4f}",
                "Specificity": f"{x['spec']:.4f}",
                "F1-score": f"{x['f1']:.4f}",
            }
        )
    out = "## 2. So sánh chế độ phát hiện\n" + md_table(pd.DataFrame(rows)) + "\n"

    out += "\n**Ma trận nhầm lẫn** (hàng = nhãn thật, cột = dự đoán)\n\n"
    for key, x in m.items():
        out += f"*{MODES[key][0]}*\n\n"
        cm = pd.DataFrame(
            {
                "": ["Thật: DROWSY", "Thật: NON_DROWSY"],
                "Dự đoán: DROWSY": [x["tp"], x["fp"]],
                "Dự đoán: NON_DROWSY": [x["fn"], x["tn"]],
            }
        )
        out += md_table(cm) + "\n\n"

    # Phân rã theo lớp
    out += "**Tỉ lệ bị dự đoán là DROWSY theo từng lớp**\n\n"
    rows = []
    for cls, label in CLASS_TO_LABEL.items():
        row = {"Lớp": cls, "Nhãn thật": label}
        for key, (name, _) in MODES.items():
            if key in preds:
                d = preds[key][preds[key]["cls"] == cls]
                if len(d):
                    row[name] = (
                        f"{pct((d['predicted_state'] == POSITIVE).mean())} ({(d['predicted_state'] == POSITIVE).sum()}/{len(d)})"
                    )
        rows.append(row)
    out += md_table(pd.DataFrame(rows)) + "\n"

    # Nhận xét tự động nếu hai chế độ cho kết quả y hệt nhau
    if len(preds) == 2:
        a, b = (preds[k] for k in MODES if k in preds)
        if (
            len(a) == len(b)
            and (a["predicted_state"].values == b["predicted_state"].values).all()
        ):
            yawn = b[b["cls"] == "Yawn"]
            note = (
                f"> ⚠️ **Lưu ý**: hai chế độ cho dự đoán giống hệt nhau trên toàn bộ {len(b)} ảnh, "
                f"tức là việc thêm MAR **chưa cải thiện** kết quả."
            )
            if len(yawn):
                note += (
                    f" Cụ thể, {len(yawn)} ảnh `Yawn` có MAR trung bình {yawn['mar'].mean():.2f} "
                    f"nhưng không ảnh nào bị gán `DROWSY`."
                )
            out += "\n" + note + "\n"
    return out


def section_latency(preds: dict) -> str:
    out = "## 3. Hiệu năng thời gian thực\n"
    summary = []
    for key, (label, _) in MODES.items():
        if key not in preds:
            continue
        d = preds[key]
        tot = d["latency_total_ms"]
        summary.append(
            {
                "Chế độ": label,
                "Độ trễ TB (ms)": f"{tot.mean():.2f}",
                "Trung vị (ms)": f"{tot.median():.2f}",
                "P95 (ms)": f"{tot.quantile(0.95):.2f}",
                "Max (ms)": f"{tot.max():.2f}",
                "FPS (1000/TB)": f"{1000 / tot.mean():.1f}",
                "FPS (1000/P95)": f"{1000 / tot.quantile(0.95):.1f}",
            }
        )
    out += md_table(pd.DataFrame(summary)) + "\n"
    out += (
        "\n*FPS ở đây chỉ tính thời gian xử lý của pipeline (face → landmark → decision), "
        "chưa gồm thời gian đọc camera/ảnh và hiển thị.*\n"
    )

    for key, (label, _) in MODES.items():
        if key not in preds:
            continue
        d = preds[key]
        total_mean = d["latency_total_ms"].mean()
        rows = []
        for name, col in LAT_COLS.items():
            rows.append(
                {
                    "Module": name,
                    "TB (ms)": f"{d[col].mean():.2f}",
                    "Trung vị (ms)": f"{d[col].median():.2f}",
                    "P95 (ms)": f"{d[col].quantile(0.95):.2f}",
                    "Max (ms)": f"{d[col].max():.2f}",
                    "Tỉ trọng": pct(d[col].mean() / total_mean),
                }
            )
        out += (
            f"\n**Phân rã độ trễ theo module — {label}**\n\n"
            + md_table(pd.DataFrame(rows))
            + "\n"
        )
    return out


def section_robustness() -> str:
    df = read_csv("robustness_results.csv")
    if df is None:
        return "## 4. Đánh giá tính bền bỉ (Robustness)\n_Không có dữ liệu `robustness_results.csv`._\n"

    base = df.iloc[0]
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "Điều kiện": r["Condition"],
                "Accuracy": f"{r['Accuracy']:.4f}",
                "Precision": f"{r['Precision']:.4f}",
                "Recall": f"{r['Recall']:.4f}",
                "F1-score": f"{r['F1_Score']:.4f}",
                "Độ trễ (ms)": f"{r['Avg_Latency_ms']:.2f}",
                "ΔF1 so với baseline": "—"
                if r.name == 0
                else f"{r['F1_Score'] - base['F1_Score']:+.4f}",
                "ΔRecall so với baseline": "—"
                if r.name == 0
                else f"{r['Recall'] - base['Recall']:+.4f}",
            }
        )
    out = (
        "## 4. Đánh giá tính bền bỉ (Robustness)\n"
        f"Dòng đầu (`{base['Condition']}`) được dùng làm mốc so sánh.\n\n"
        + md_table(pd.DataFrame(rows))
        + "\n"
    )

    worst = df.iloc[1:].sort_values("F1_Score").iloc[0] if len(df) > 1 else None
    if worst is not None:
        out += (
            f"\n- Điều kiện làm giảm hiệu năng nhiều nhất: **{worst['Condition']}** "
            f"(F1 {worst['F1_Score']:.4f}, giảm {base['F1_Score'] - worst['F1_Score']:.4f}; "
            f"Recall giảm {base['Recall'] - worst['Recall']:.4f}).\n"
        )
    lat_range = df["Avg_Latency_ms"].max() - df["Avg_Latency_ms"].min()
    out += f"- Độ trễ gần như không đổi giữa các điều kiện (chênh lệch tối đa {lat_range:.2f} ms).\n"
    return out


def section_by_subject(preds: dict) -> str:
    key = "eye_yawn" if "eye_yawn" in preds else next(iter(preds))
    d = preds[key]
    rows = []
    for s, g in d.groupby("subject"):
        x = metrics(g)
        rows.append(
            {
                "Subject": s,
                "Số ảnh": x["n"],
                "Accuracy": f"{x['acc']:.4f}",
                "Precision": f"{x['prec']:.4f}",
                "Recall": f"{x['rec']:.4f}",
                "F1-score": f"{x['f1']:.4f}",
            }
        )
    return (
        f"## 5. Kết quả theo từng đối tượng ({MODES[key][0]})\n"
        + md_table(pd.DataFrame(rows))
        + "\n"
    )


def section_conclusion(preds: dict) -> str:
    key = "eye_yawn" if "eye_yawn" in preds else next(iter(preds))
    x, d = metrics(preds[key]), preds[key]
    fps = 1000 / d["latency_total_ms"].mean()
    bullets = [
        f"Độ chính xác tổng thể {pct(x['acc'])}; Precision cao ({pct(x['prec'])}) nhưng Recall thấp "
        f"({pct(x['rec'])}) — hệ thống **ít báo động nhầm nhưng bỏ sót nhiều trường hợp buồn ngủ** "
        f"({x['fn']}/{x['tp'] + x['fn']} ca).",
        f"Pipeline chạy khoảng {fps:.0f} FPS (độ trễ TB {d['latency_total_ms'].mean():.1f} ms), "
        f"{'đáp ứng' if fps >= 15 else 'chưa đáp ứng'} ngưỡng thời gian thực 15 FPS.",
        f"Bottleneck nằm ở module landmark ({pct(d['latency_landmark_ms'].mean() / d['latency_total_ms'].mean())} thời gian); "
        f"Decision Engine gần như không đáng kể.",
    ]
    return "## 6. Kết luận\n" + "\n".join(f"- {b}" for b in bullets) + "\n"


# ---------------------------------------------------------------- Main
def generate_benchmark_report():
    preds = {}
    for key, (_, fname) in MODES.items():
        df = load_predictions(fname)
        if df is not None:
            preds[key] = df
    if not preds:
        print("Lỗi: không có file raw_predictions nào để tổng hợp.")
        return

    md = "\n".join(
        [
            "# Báo cáo Benchmark Hệ thống Phát hiện Buồn ngủ\n",
            f"_Tạo tự động từ dữ liệu trong `{RESULTS_DIR.name}/`_\n",
            section_overview(preds),
            section_mode_comparison(preds),
            section_latency(preds),
            section_robustness(),
            section_by_subject(preds),
            section_conclusion(preds),
        ]
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(md, encoding="utf-8")
    print(f"[*] Đã tạo báo cáo benchmark: {OUTPUT_FILE}")

    # Đối chiếu với comparison_report.csv để phát hiện sai lệch số liệu
    cmp = read_csv("comparison_report.csv")
    if cmp is not None:
        for _, r in cmp.iterrows():
            if r["Mode"] in preds:
                x = metrics(preds[r["Mode"]])
                if (
                    abs(x["f1"] - r["F1_Score"]) > 1e-3
                    or abs(x["acc"] - r["Accuracy"]) > 1e-3
                ):
                    print(
                        f"[!] Số liệu tính lại của '{r['Mode']}' lệch so với comparison_report.csv "
                        f"(kiểm tra CLASS_TO_LABEL)."
                    )


if __name__ == "__main__":
    generate_benchmark_report()
