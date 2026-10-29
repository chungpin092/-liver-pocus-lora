"""
彙整 anomaly_map.py 的結果，計算每個 strength 的指標。

預期資料夾結構（run_anomaly.bat 會自動產生）：
  results/s0.4/hemorrhage/scores.csv
  results/s0.4/normal/scores.csv

輸出（可分享）：
  results/summary.csv          每個 strength 的影像層級 AUROC、像素層級 AUROC、Dice
  results/strength_curve.png   strength 對 AUROC 的曲線
"""
import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pixel_metrics(group_dir, top_percent=5.0):
    """有標註的病例才算：像素層級 AUROC 與 Dice（取最亮前 top_percent% 像素當預測）。"""
    id_map_path = os.path.join(group_dir, "id_map.csv")
    if not os.path.exists(id_map_path):
        return None, None
    scores, labels, dices = [], [], []
    for row in read_csv(id_map_path):
        if not row["mask_file"] or not os.path.exists(row["mask_file"]):
            continue
        amap = np.load(os.path.join(group_dir, f"{row['case']}_amap.npy")).astype(np.float32)
        gt = np.asarray(Image.open(row["mask_file"]).convert("L").resize((512, 512))) > 127
        scores.append(amap.ravel())
        labels.append(gt.ravel())
        pred = amap > np.percentile(amap, 100 - top_percent)
        dices.append(2 * (pred & gt).sum() / (pred.sum() + gt.sum() + 1e-6))
    if not dices:
        return None, None
    labels = np.concatenate(labels)
    if labels.min() == labels.max():
        return None, float(np.mean(dices))
    return float(roc_auc_score(labels, np.concatenate(scores))), float(np.mean(dices))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    args = p.parse_args()

    rows = []
    for s_dir in sorted(glob.glob(os.path.join(args.results_dir, "s*"))):
        h_csv = os.path.join(s_dir, "hemorrhage", "scores.csv")
        n_csv = os.path.join(s_dir, "normal", "scores.csv")
        if not (os.path.exists(h_csv) and os.path.exists(n_csv)):
            continue
        strength = float(os.path.basename(s_dir)[1:])
        h = [float(r["score"]) for r in read_csv(h_csv)]
        n = [float(r["score"]) for r in read_csv(n_csv)]
        img_auroc = roc_auc_score([1] * len(h) + [0] * len(n), h + n)
        pix_auroc, dice = pixel_metrics(os.path.join(s_dir, "hemorrhage"))
        rows.append({
            "strength": strength,
            "n_hemorrhage": len(h),
            "n_normal": len(n),
            "mean_score_hemorrhage": f"{np.mean(h):.5f}",
            "mean_score_normal": f"{np.mean(n):.5f}",
            "image_auroc": f"{img_auroc:.4f}",
            "pixel_auroc": "" if pix_auroc is None else f"{pix_auroc:.4f}",
            "dice_top5pct": "" if dice is None else f"{dice:.4f}",
        })

    if not rows:
        raise SystemExit(f"{args.results_dir} 裡找不到 s*/hemorrhage 與 s*/normal 的 scores.csv")

    rows.sort(key=lambda r: r["strength"])
    with open(os.path.join(args.results_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    xs = [r["strength"] for r in rows]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(xs, [float(r["image_auroc"]) for r in rows], marker="o", label="Image-level AUROC")
    if all(r["pixel_auroc"] for r in rows):
        ax.plot(xs, [float(r["pixel_auroc"]) for r in rows], marker="s", label="Pixel-level AUROC")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
    ax.set_xlabel("strength")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, "strength_curve.png"), dpi=150)
    plt.close(fig)

    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
