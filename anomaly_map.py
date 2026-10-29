"""
用肝臟 POCUS LoRA 把影像「修復成正常」，計算差異圖找出異常（出血）區域。
原理與參數說明見 HEMORRHAGE_DIFF_MAP.md。

輸出分成兩類：
  可分享（會被 git 追蹤）：scores.csv、case_XXX_heatmap.png（只有熱圖，不含原圖）
  僅本機（被 .gitignore 擋掉）：*_panel.png（含原圖）、*_amap.npy、id_map.csv（原始檔名對照）

用法：
  python anomaly_map.py --input_dir data/hemorrhage --output_dir results/s0.4/hemorrhage --strength 0.4
"""
import argparse
import csv
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 若 runwayml 的路徑下載不到，可改成 "stable-diffusion-v1-5/stable-diffusion-v1-5"
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
PROMPT = "a photo of liver, normal healthy liver, uniform echogenicity, smooth homogenous texture, clear liver margins, high quality, highly detailed medical ultrasound image"
NEGATIVE_PROMPT = "hemorrhage, bleeding, cysts, tumors, lesions, shadows, medical anomaly, blurry, text, artifacts, low resolution"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def load_pipe(lora_path):
    import torch
    from diffusers import StableDiffusionImg2ImgPipeline

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, safety_checker=None
    )
    pipe.load_lora_weights(lora_path)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to("cuda")


def to_gray(img):
    return np.asarray(img.convert("L"), dtype=np.float32) / 255.0


def vae_roundtrip(pipe, img):
    """原圖只經過 VAE 編碼再解碼（不去噪）。
    VAE 本身會讓 speckle 變模糊，拿這張當比較基準，才不會把 VAE 的失真誤判成異常。"""
    import torch

    with torch.no_grad():
        x = pipe.image_processor.preprocess(img).to("cuda", torch.float16)
        latents = pipe.vae.encode(x).latent_dist.mean
        y = pipe.vae.decode(latents).sample
    return pipe.image_processor.postprocess(y, output_type="pil")[0]


def repair(pipe, img, strength, seed):
    """img2img：加 strength 比例的雜訊後，用正常肝臟 prompt 去噪。"""
    import torch

    generator = torch.Generator(device="cuda").manual_seed(seed)
    return pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=img,
        strength=strength,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=generator,
    ).images[0]


def sector_mask(img, erode_px=15):
    """取超音波扇形區域（非黑色背景的凸包），扇形外不計分。
    凸包可避免把扇形內的無回音液體（黑色）也排除掉。"""
    g = np.asarray(img.convert("L"))
    pts = cv2.findNonZero((g > 10).astype(np.uint8))
    mask = np.zeros_like(g, dtype=np.uint8)
    if pts is not None:
        cv2.fillConvexPoly(mask, cv2.convexHull(pts), 1)
        mask = cv2.erode(mask, np.ones((erode_px, erode_px), np.uint8))
    return mask.astype(np.float32)


def anomaly_map(pipe, img, strength, seeds, blur_sigma):
    ref = to_gray(vae_roundtrip(pipe, img))
    repaired_imgs, signed_diffs = [], []
    for s in seeds:
        rep = repair(pipe, img, strength, s)
        repaired_imgs.append(rep)
        # 正值 = 修復後比原圖亮 = 原圖該處偏暗（低回音／無回音，常見於液體、血腫）
        signed_diffs.append(to_gray(rep) - ref)
    # 多個 seed 平均，壓掉隨機 speckle 雜訊
    abs_map = cv2.GaussianBlur(np.mean(np.abs(signed_diffs), axis=0), (0, 0), blur_sigma)
    mask = sector_mask(img)
    return abs_map * mask, repaired_imgs[0], mask


def save_heatmap(path, amap, vmax):
    """只存熱圖本身（不疊原圖），可安全上傳到公開 repo。"""
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(amap, cmap="jet", vmin=0, vmax=vmax)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_panel(path, img, repaired, amap, vmax, gt_mask=None):
    """含原圖的四／五格圖，只留在本機。"""
    n = 5 if gt_mask is not None else 4
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    gray = to_gray(img)
    axes[0].imshow(gray, cmap="gray"); axes[0].set_title("Original")
    axes[1].imshow(to_gray(repaired), cmap="gray"); axes[1].set_title("Repaired (normal)")
    axes[2].imshow(amap, cmap="jet", vmin=0, vmax=vmax); axes[2].set_title("Anomaly map")
    axes[3].imshow(gray, cmap="gray")
    axes[3].imshow(amap, cmap="jet", alpha=0.45, vmin=0, vmax=vmax); axes[3].set_title("Overlay")
    if gt_mask is not None:
        axes[4].imshow(gray, cmap="gray")
        axes[4].contour(gt_mask, levels=[0.5], colors="lime"); axes[4].set_title("Ground truth")
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def find_mask(mask_dir, stem):
    if not mask_dir:
        return None
    for ext in IMAGE_EXTS:
        p = os.path.join(mask_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mask_dir", default=None, help="標註資料夾，檔名（不含副檔名）需與影像相同，白=出血")
    p.add_argument("--lora", default="./pytorch_lora_weights.safetensors")
    p.add_argument("--strength", type=float, default=0.4)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--blur_sigma", type=float, default=4.0)
    p.add_argument("--vmax", type=float, default=0.15, help="熱圖色階上限，所有影像共用才能互相比較")
    args = p.parse_args()

    names = sorted(n for n in os.listdir(args.input_dir) if n.lower().endswith(IMAGE_EXTS))
    if not names:
        raise SystemExit(f"{args.input_dir} 裡找不到影像")

    os.makedirs(args.output_dir, exist_ok=True)
    pipe = load_pipe(args.lora)
    seeds = list(range(args.n_seeds))

    rows, id_rows = [], []
    for i, name in enumerate(names, start=1):
        # 對外一律用 case_001 這類編號，避免檔名洩漏病歷號
        case_id = f"case_{i:03d}"
        stem = os.path.splitext(name)[0]
        img = Image.open(os.path.join(args.input_dir, name)).convert("RGB").resize((512, 512))
        amap, repaired, mask = anomaly_map(pipe, img, args.strength, seeds, args.blur_sigma)

        mask_path = find_mask(args.mask_dir, stem)
        gt = None
        if mask_path:
            gt = (to_gray(Image.open(mask_path).resize((512, 512))) > 0.5).astype(np.float32)

        np.save(os.path.join(args.output_dir, f"{case_id}_amap.npy"), amap.astype(np.float16))
        save_heatmap(os.path.join(args.output_dir, f"{case_id}_heatmap.png"), amap, args.vmax)
        save_panel(os.path.join(args.output_dir, f"{case_id}_panel.png"), img, repaired, amap, args.vmax, gt)

        inside = amap[mask > 0]
        # 影像層級分數：扇形內差異值的第 99 百分位（比 max 穩定，不怕單一雜訊點）
        score = float(np.percentile(inside, 99)) if inside.size else 0.0
        rows.append({"case": case_id, "score": f"{score:.6f}", "has_mask": int(gt is not None)})
        id_rows.append({"case": case_id, "original_file": name, "mask_file": mask_path or ""})
        print(f"[{i}/{len(names)}] {case_id}: score={score:.4f}")

    with open(os.path.join(args.output_dir, "scores.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case", "score", "has_mask"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.output_dir, "id_map.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case", "original_file", "mask_file"])
        w.writeheader()
        w.writerows(id_rows)


if __name__ == "__main__":
    main()
