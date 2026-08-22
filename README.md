# 小型資料包 —— 跑通垃圾偵測訓練流程

2026-08-20 產出。**用途只有一個：讓接手的人在自己的機器上把整條 training pipeline 跑起來。**
不是拿來訓練正式模型，也不是拿來產生對外報告的數字（原因見下方「兩個限制」）。

完整的資料規格、模型版本紀錄、已知問題請看 `handoff_20260820_training/`。


> **這個 repo 只有程式碼與清單（1.2 MB）。** 影像（`dataset/`，5 GB）與權重（`yolo11l.pt`、`yolo26n.pt`）
> 太大不進 git，請跟負責人拿完整的 `handoff_smoke_20260820/`，解開後用 `SHA256SUMS` 驗證：
>
> ```bash
> sha256sum -c SHA256SUMS | grep -v ': OK$'     # macOS: shasum -a 256 -c
> ```
> 沒有輸出就是 6,487 個檔全對。拿到完整包後，下面的說明才適用。

---

## 內容物

| 檔案 | 說明 | 大小 |
|---|---|---:|
| `dataset/` | 等比例抽樣的子集，YOLO 格式（`images/` 與 `labels/` 鏡像樹） | 5.1 GB |
| `dataset/data.yaml` | Ultralytics 設定檔，`nc: 1`, `names: {0: trash}` | — |
| `train.py` | **可直接執行的訓練腳本**，53 行，不綁 W&B | 2 KB |
| `build_subset.py` | 產生這個子集的腳本，可重跑、可改容量 | 7 KB |
| `SUBSET_MANIFEST.csv` | 抽了哪 3,239 張的完整清單 | 200 KB |
| `yolo11l.pt` | 起始權重（COCO 預訓練）。已附上，離線也能開跑 | 51 MB |
| `yolo26n.pt` | Ultralytics 做 AMP 檢查時會自動抓的小模型，一併附上 | 5.5 MB |

---

## 三步驟上手

```bash
pip install ultralytics

# 1) 跑通流程：1 個 epoch、低解析度，確認環境沒問題
python3 train.py --epochs 1 --imgsz 640 --name smoke

# 2) 正式跑（專案 v4 配方：imgsz 1280 / batch 8 / patience 20）
python3 train.py --name my_run
```

`train.py` 只有 53 行，核心就兩件事：

```python
model = YOLO("yolo11l.pt")   # ← 載入 COCO 預訓練權重
model.train(data=..., epochs=..., imgsz=1280, batch=8, patience=20)
```

⚠️ `model.train()` 裡的 `pretrained=True/False` 在偵測訓練中是**無效參數**。
決定要不要用預訓練的是 `YOLO()` 那一行：`.pt` = 預訓練，`.yaml` = 從零開始。

輸出在 `runs/detect/trash_detect/<name>/`：`weights/best.pt`、`weights/last.pt`、
`results.csv`（每個 epoch 的指標）、以及 PR / F1 曲線圖。

起始權重 `yolo11l.pt` 已附在包裡（在這個目錄下執行就會直接用它，不需連網）。
**已實測**：2026-08-21 在 RTX 5090 上跑 `--epochs 1 --imgsz 640 --batch 4`，約 30 秒完成，
`weights/best.pt` 與 `results.csv` 正常產出。已驗證確實載入 COCO 預訓練權重
（訓練後各層與基準權重的 cosine similarity 0.97–0.9997；隨機初始化的話會是 ~0）。
沒有 GPU 就加 `--device cpu`；VRAM 不夠就調小 `--batch`。
`data.yaml` 裡的絕對路徑會在每次啟動時自動對齊到當前位置，**換機器不用手動改**。

---

## 這份資料是怎麼抽的

來源：`final_dataset_labeled_only`（15,810 幀 / 24.4 GiB，全部都是有標註的正樣本）。

| | 來源 | 這份子集 | 比例 |
|---|---:|---:|---:|
| 資料夾 | 61 | **61（全數保留）** | 100% |
| train | 9,730 幀 | 1,991 幀 | 20.5% |
| validation | 2,846 幀 | 586 幀 | 20.6% |
| test | 3,234 幀 | 662 幀 | 20.5% |
| **合計** | **15,810 幀** | **3,239 幀 / 5,133 框** | **20.4%** |

抽樣規則有三條，都是為了讓小子集仍然「像」原資料：

1. **不丟資料夾**，只在資料夾內部抽 —— 丟掉整個資料夾等於丟掉整個場景。
2. 資料夾內用**均勻時間間隔**抽（不是取前 N 張）—— 同一個資料夾是一段連續影片，
   取前 N 張只會拿到影片開頭幾秒。
3. 比例由**目標容量反推**（二分搜尋實際位元組數），不是用幀數硬算 ——
   各資料夾檔案大小不一，幀數 20% 不等於容量 20%。

要換容量重做：

```bash
rm -rf dataset && python3 build_subset.py --target-gib 2.0
python3 build_subset.py --dry-run --target-gib 2.0   # 只試算不複製
```

---

## 要寄給別人

```bash
cd /home/weihong
tar czf handoff_smoke_20260820.tar.gz handoff_smoke_20260820/
```

壓縮後約 4.8 GB（PNG 已經是壓縮格式，tar.gz 幫助有限）。
若要更小，改用 `--target-gib 2.0` 重做子集，或改用 JPEG 版來源
（`/mnt/backups/weihong/final_dataset_labeled_only_jpg`，q95，全量只有 6.0 GB）。
