# 用「正常肝臟生成模型」找出血點：差異圖（Anomaly Map）作法說明

> 適用於本 repo 的 `pytorch_lora_weights.safetensors`（SD 1.5 + 肝臟 POCUS LoRA，只用正常影像訓練）。
> 核心想法：**模型只會畫正常肝臟 → 把有出血的影像交給它「修復」→ 修復前後差最多的地方，就是不正常的地方。**

---

## 1. 原理

```
            加入部分雜訊                 用 "normal healthy liver" 去噪
有出血的原圖 ───────────────▶ 模糊的影像 ─────────────────────────────▶ 「修復成正常」的影像
     │                                                                     │
     └───────────────────────────── 相減 ───────────────────────────────────┘
                                     │
                                     ▼
                       差異圖（亮的地方 = 可能出血位置）
```

- 模型在訓練時**沒看過出血**，所以去噪時只能把那個區域畫成它認識的「正常肝實質」。
- 正常的區域（肝實質、血管、邊界）模型畫得出來，修復前後幾乎一樣，差異小。
- 出血區域被「改掉」，差異大，在差異圖上會亮起來。
- **出血影像只拿來測試，不拿來訓練**，所以只要幾十張就能做出結果。

這個方法在文獻中稱為**以擴散模型做無監督異常偵測（diffusion-based unsupervised anomaly detection）**，可引用：

- Wolleb et al., *Diffusion Models for Medical Anomaly Detection*, MICCAI 2022
- Pinaya et al., *Fast Unsupervised Brain Anomaly Detection and Segmentation with Diffusion Models*, MICCAI 2022
- Wyatt et al., *AnoDDPM: Anomaly Detection with Denoising Diffusion Probabilistic Models using Simplex Noise*, CVPR Workshops 2022

---

## 2. 需要準備的資料

| 資料 | 用途 | 數量建議 |
|---|---|---|
| 有出血的 POCUS 影像 | 測試：看差異圖是否亮在出血處 | 20–50 張以上 |
| **沒參與訓練**的正常影像 | 對照組：正常影像的差異圖應該很暗 | 與出血影像數量相近 |
| 出血位置標註（mask，可選） | 算像素級指標（Dice、AUROC） | 請醫師/技師圈出血範圍 |

注意：
- 正常對照組**一定不能是訓練集裡的影像**，否則模型修得特別好，結果會偏樂觀。
- 影像上的文字、刻度、機器 logo 先裁掉或塗黑，不然會被當成異常。

---

## 3. 程式碼（`anomaly_map.py`）

```python
"""
用肝臟 POCUS LoRA 把影像「修復成正常」，計算差異圖找出異常（出血）區域。

用法：
  python anomaly_map.py --input_dir data/hemorrhage --output_dir out/hemorrhage
  python anomaly_map.py --input_dir data/normal_test --output_dir out/normal_test
"""
import argparse
import csv
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

# 若 runwayml 的路徑下載不到，可改成 "stable-diffusion-v1-5/stable-diffusion-v1-5"
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
PROMPT = "a photo of liver, normal healthy liver, uniform echogenicity, smooth homogenous texture, clear liver margins, high quality, highly detailed medical ultrasound image"
NEGATIVE_PROMPT = "hemorrhage, bleeding, cysts, tumors, lesions, shadows, medical anomaly, blurry, text, artifacts, low resolution"


def load_pipe(lora_path):
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, safety_checker=None
    )
    pipe.load_lora_weights(lora_path)
    return pipe.to("cuda")


def to_gray(img):
    return np.asarray(img.convert("L"), dtype=np.float32) / 255.0


@torch.no_grad()
def vae_roundtrip(pipe, img):
    """原圖只經過 VAE 編碼再解碼（不去噪）。
    VAE 本身會讓 speckle 變模糊，拿這張當比較基準，才不會把 VAE 的失真誤判成異常。"""
    x = pipe.image_processor.preprocess(img).to("cuda", torch.float16)
    latents = pipe.vae.encode(x).latent_dist.mean
    y = pipe.vae.decode(latents).sample
    return pipe.image_processor.postprocess(y, output_type="pil")[0]


def repair(pipe, img, strength, seed):
    """img2img：加 strength 比例的雜訊後，用正常肝臟 prompt 去噪。"""
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
    signed = np.mean(signed_diffs, axis=0)  # 多個 seed 平均，壓掉隨機 speckle 雜訊
    abs_map = np.mean(np.abs(signed_diffs), axis=0)
    abs_map = cv2.GaussianBlur(abs_map, (0, 0), blur_sigma)
    signed = cv2.GaussianBlur(signed, (0, 0), blur_sigma)
    mask = sector_mask(img)
    return abs_map * mask, signed * mask, repaired_imgs[0], mask


def save_figure(path, img, repaired, amap, gt_mask=None):
    n = 5 if gt_mask is not None else 4
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    gray = to_gray(img)
    vmax = np.percentile(amap, 99.5) + 1e-6
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mask_dir", default=None, help="標註資料夾，檔名需與影像相同（白=出血）")
    p.add_argument("--lora", default="./pytorch_lora_weights.safetensors")
    p.add_argument("--strength", type=float, default=0.4)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--blur_sigma", type=float, default=4.0)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pipe = load_pipe(args.lora)
    seeds = list(range(args.n_seeds))

    rows = []
    for name in sorted(os.listdir(args.input_dir)):
        if not name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue
        img = Image.open(os.path.join(args.input_dir, name)).convert("RGB").resize((512, 512))
        amap, signed, repaired, mask = anomaly_map(pipe, img, args.strength, seeds, args.blur_sigma)

        gt = None
        if args.mask_dir and os.path.exists(os.path.join(args.mask_dir, name)):
            gt = (to_gray(Image.open(os.path.join(args.mask_dir, name)).resize((512, 512))) > 0.5).astype(np.float32)

        stem = os.path.splitext(name)[0]
        np.save(os.path.join(args.output_dir, f"{stem}_amap.npy"), amap)
        save_figure(os.path.join(args.output_dir, f"{stem}_panel.png"), img, repaired, amap, gt)

        inside = amap[mask > 0]
        # 影像層級分數：扇形內差異值的第 99 百分位（比 max 穩定，不怕單一雜訊點）
        score = float(np.percentile(inside, 99)) if inside.size else 0.0
        rows.append({"image": name, "score": score})
        print(f"{name}: score={score:.4f}")

    with open(os.path.join(args.output_dir, "scores.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "score"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
```

安裝：

```bash
pip install diffusers transformers accelerate torch opencv-python matplotlib scikit-learn
```

輸出（每張影像）：
- `xxx_panel.png`：原圖｜修復後｜差異熱圖｜疊圖｜（醫師標註）——**這張就是報告裡的主圖**
- `xxx_amap.npy`：差異圖數值，給後面算指標用
- `scores.csv`：每張影像的異常分數

---

## 4. 怎麼看差異圖

| 你看到的 | 代表 |
|---|---|
| 熱圖紅/黃區塊落在醫師標註的出血位置 | ✅ 方法有效 |
| 正常對照組的熱圖整張偏藍、沒有明顯熱點 | ✅ 偽陽性低 |
| 熱點落在膽囊、大血管、腎臟 | ⚠️ 模型對這些正常的暗區結構畫得不夠好（見第 6 節） |
| 整張圖都是細碎的紅點 | ⚠️ speckle 雜訊沒壓掉：調高 `--n_seeds` 或 `--blur_sigma` |
| 整張圖都很紅、包括正常組織 | ⚠️ `strength` 太高，模型把整個解剖結構都改掉了 |
| 出血處沒亮 | ⚠️ `strength` 太低，雜訊不夠，模型直接把出血「照抄」回來 |

**有號差異圖（`signed`）的用法：** 正值代表「原圖比正常還暗」。超音波上腹腔游離液體、較久的血腫多呈**低回音／無回音（黑）**，看正值區域更準。但**急性出血可能是等回音或高回音**，所以主要還是看絕對差異圖。

---

## 5. 最重要的參數：`strength`

`strength` 控制加多少雜訊（0 = 不改，1 = 完全重畫）。

- 太低（< 0.3）：出血還看得出來，模型照著畫回去 → 差異圖抓不到
- 太高（> 0.7）：模型連正常結構都重畫 → 整張都是差異
- **建議先掃 `0.2, 0.3, 0.4, 0.5, 0.6`**，每個值都跑同一組出血影像加正常影像，選「出血組與正常組分數差最大」的值

```bash
for s in 0.2 0.3 0.4 0.5 0.6; do
  python anomaly_map.py --input_dir data/hemorrhage   --output_dir out/s$s/hemorrhage --strength $s
  python anomaly_map.py --input_dir data/normal_test  --output_dir out/s$s/normal     --strength $s
done
```

這個掃描結果本身就可以畫成一張圖放進報告（x 軸 strength，y 軸 AUROC）。

---

## 6. 評估指標（讓結果有數字可報）

### 6.1 影像層級：能不能分辨「有出血 vs 正常」（不需要標註）

```python
import pandas as pd
from sklearn.metrics import roc_auc_score

h = pd.read_csv("out/s0.4/hemorrhage/scores.csv")
n = pd.read_csv("out/s0.4/normal/scores.csv")
y_true = [1] * len(h) + [0] * len(n)
y_score = list(h["score"]) + list(n["score"])
print("Image-level AUROC:", roc_auc_score(y_true, y_score))
```

### 6.2 像素層級：熱點位置對不對（需要醫師標註 mask）

```python
import os
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

amap_dir, mask_dir = "out/s0.4/hemorrhage", "data/hemorrhage_masks"
all_scores, all_labels, dices = [], [], []
for f in os.listdir(amap_dir):
    if not f.endswith("_amap.npy"):
        continue
    stem = f[:-len("_amap.npy")]
    amap = np.load(os.path.join(amap_dir, f))
    gt = np.asarray(Image.open(os.path.join(mask_dir, stem + ".png")).convert("L").resize((512, 512))) > 127
    all_scores.append(amap.ravel()); all_labels.append(gt.ravel())
    pred = amap > np.percentile(amap, 95)  # 取前 5% 最亮的像素當預測區域
    dices.append(2 * (pred & gt).sum() / (pred.sum() + gt.sum() + 1e-6))

print("Pixel-level AUROC:", roc_auc_score(np.concatenate(all_labels), np.concatenate(all_scores)))
print("Mean Dice:", np.mean(dices))
```

報告時的參考解讀：影像層級 AUROC > 0.7 即有初步可行性；像素層級 Dice 在無監督方法中通常偏低（0.2–0.4 很常見），重點是和「隨機猜」「單純找最暗區域」等 baseline 比。

### 6.3 建議的 baseline（證明不是只靠「找黑的地方」）

| Baseline | 作法 |
|---|---|
| 強度門檻 | 直接把原圖最暗的 5% 像素當預測 |
| 只用 VAE 重建誤差 | `strength=0`，只比 VAE 編碼解碼前後差異 |
| 本方法 | LoRA img2img 修復 + 多 seed 平均 |

本方法要贏過前兩者，才能說是「生成模型學到的正常外觀」在發揮作用。

---

## 7. 已知限制（報告中要誠實寫出）

1. **正常的暗區結構會被誤判。** 膽囊、下腔靜脈、門靜脈、腎臟也是黑的。如果模型沒把它們學好，就會變成偽陽性。可以統計偽陽性落在哪些結構，當作分析的一部分。
2. **急性出血不一定是黑的。** 急性期可能是等回音或高回音，和肝實質不易區分，這時差異圖也會比較弱。
3. **Morison's pouch 的游離液體在肝臟之外。** 如果訓練資料的視野主要在肝實質，模型對肝腎交界的畫法可能本來就不穩。要看訓練影像涵蓋的視野。
4. **隨機性。** img2img 每次結果不同，所以用多個 seed 平均；報告中要寫明 seed 數量。
5. **512×512 且經過 VAE 壓縮，** 很小的出血點（< 10 px）可能抓不到。

---

## 8. 報告／海報可用的呈現方式

1. **主圖：** 3–4 個案例的 `panel.png` 排成一列（出血成功、出血失敗、正常對照各至少一例），**失敗案例也要放**。
2. **Strength 掃描曲線：** strength 對 AUROC。
3. **指標表：** 本方法 vs. 兩個 baseline 的影像層級 AUROC、像素層級 AUROC、Dice。
4. **一句話賣點：**
   > 「我們只用正常肝臟超音波訓練生成模型，不需要任何出血資料，就能在測試時標示出可能的出血位置。」
