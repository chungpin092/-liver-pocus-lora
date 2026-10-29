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

## 3. 程式與一鍵執行（實驗室 Windows 電腦）

repo 裡的三個檔案：

| 檔案 | 作用 |
|---|---|
| `anomaly_map.py` | 對一個資料夾的影像做修復、算差異圖與分數 |
| `evaluate_anomaly.py` | 彙整所有 strength 的結果，算 AUROC / Dice，畫 strength 曲線 |
| `run_anomaly.bat` | 一鍵跑完：出血組＋正常組 × strength 0.2–0.6，最後自動評估 |

### 步驟

```powershell
# 1. 第一次：下載 repo（HTTPS 或 SSH 擇一）
git clone -b claude/hemorrhage-diff-map https://github.com/chungpin092/-liver-pocus-lora.git
# git clone -b claude/hemorrhage-diff-map git@github.com:chungpin092/-liver-pocus-lora.git

# 之後更新
git pull

# 2. 安裝套件（已有 torch / diffusers 可略過前兩個）
pip install diffusers transformers accelerate torch opencv-python matplotlib scikit-learn
```

3. 用記事本打開 `run_anomaly.bat`，改最上面三個路徑：`HEMO_DIR`（出血影像）、`NORMAL_DIR`（沒參與訓練的正常影像）、`MASK_DIR`（標註，可留空）。
4. 雙擊 `run_anomaly.bat`。每張影像 × 5 個 seed × 5 個 strength，時間較長，可先把 `STRENGTHS` 改成 `0.4` 試跑。
5. 跑完後上傳可分享的結果：

```powershell
git add results
git status        # 確認清單裡「沒有」*_panel.png、*.npy、id_map.csv
git commit -m "Add anomaly map results"
git push
```

### 輸出與隱私

所有對外檔名一律改成 `case_001`、`case_002`…，避免原始檔名洩漏病歷號。

| 檔案 | 內容 | 會上傳 GitHub？ |
|---|---|---|
| `scores.csv` | 每個 case 的異常分數 | ✅ |
| `case_XXX_heatmap.png` | **只有**差異熱圖，不含原圖 | ✅ |
| `results/summary.csv`、`strength_curve.png` | 各 strength 的指標與曲線 | ✅ |
| `case_XXX_panel.png` | 原圖｜修復後｜熱圖｜疊圖｜標註（**報告主圖**） | ❌ 只留本機 |
| `case_XXX_amap.npy` | 差異圖數值 | ❌ 只留本機 |
| `id_map.csv` | case 編號 ↔ 原始檔名對照 | ❌ 只留本機 |

`.gitignore` 已設定成 `results/` 底下預設全部不上傳，只放行上表打勾的檔案。**這個 repo 是公開的，原始影像與 panel 不要用 `git add -f` 硬加。**

### 參數

| 參數 | 預設 | 說明 |
|---|---|---|
| `--strength` | 0.4 | 加多少雜訊，見第 5 節 |
| `--n_seeds` | 5 | 每張修復幾次取平均，越多雜訊越少、越慢 |
| `--blur_sigma` | 4.0 | 差異圖模糊程度，壓 speckle |
| `--vmax` | 0.15 | 熱圖色階上限；所有影像共用才能互相比較。熱圖全紅就調高，全藍就調低 |

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

**注意回音特性：** 腹腔游離液體、較久的血腫多呈低回音／無回音（黑），差異圖最容易抓到；**急性出血可能是等回音或高回音**，差異會比較弱，判讀時要分開看。

---

## 5. 最重要的參數：`strength`

`strength` 控制加多少雜訊（0 = 不改，1 = 完全重畫）。

- 太低（< 0.3）：出血還看得出來，模型照著畫回去 → 差異圖抓不到
- 太高（> 0.7）：模型連正常結構都重畫 → 整張都是差異
- **建議先掃 `0.2, 0.3, 0.4, 0.5, 0.6`**，每個值都跑同一組出血影像加正常影像，選「出血組與正常組分數差最大」的值

`run_anomaly.bat` 預設就會掃 `0.2 0.3 0.4 0.5 0.6`，跑完後 `results/summary.csv` 與 `results/strength_curve.png` 會列出每個值的結果。

這個掃描結果本身就可以畫成一張圖放進報告（x 軸 strength，y 軸 AUROC）。

---

## 6. 評估指標（讓結果有數字可報）

`evaluate_anomaly.py`（bat 會自動執行）已包含下面兩項計算，這裡列出原理。

### 6.1 影像層級：能不能分辨「有出血 vs 正常」（不需要標註）

每張影像的分數 = 扇形內差異值的第 99 百分位。出血組標 1、正常組標 0，計算 **AUROC**（`summary.csv` 的 `image_auroc`）。

### 6.2 像素層級：熱點位置對不對（需要醫師標註 mask）

- **像素層級 AUROC**（`pixel_auroc`）：所有有標註的出血影像，每個像素的差異值 vs. 是否在標註內。
- **Dice**（`dice_top5pct`）：每張取差異最大的前 5% 像素當預測區域，和標註比重疊程度，再取平均。

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
