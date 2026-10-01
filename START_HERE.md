# 垃圾偵測專案 —— 交接

用 YOLO11l 逐幀偵測監視器畫面裡的垃圾（單類別 `trash`），輸出 bounding box。
第二階段（把垃圾跟附近的人／車配對、判定亂丟）不在這份交接的範圍。

順序：先跑小資料包 → 資料集 → 訓練 → 推論與測試。

環境：在這台伺服器上先 `source /home/weihong/.venv/bin/activate`（Ultralytics 8.4.41）；
在自己的電腦上 `pip install ultralytics`。

---

## 1. 先跑 5 GB 小資料包

`handoff_smoke_20260820/` 是完整資料的縮小版，用來先把整條訓練流程跑通。

```bash
cd /home/weihong/handoff_smoke_20260820
python3 train.py --epochs 1 --imgsz 640 --name smoke
```

跑完在 `runs/detect/trash_detect/smoke/` 會有 `weights/best.pt`、`weights/last.pt`、`results.csv`。

- **內容**：3,239 幀 / 5,133 框，61 個影片資料夾全部保留，每個資料夾內等間隔抽約 20%。
  起始權重 `yolo11l.pt` 已附在包裡。
- **限制**：它抽自 `final_dataset_labeled_only`，**沒有背景幀**，量不到背景上的誤報，
  precision 和 F1 會虛高。只用來確認流程能跑，不用來比較模型。
- **標完新資料之後**：CVAT 匯出的是 3 類（0 Person / 1 Vehicle / 2 Trash），訓練前用
  `python3 filter_trash_labels.py --src <CVAT匯出資料夾> --out <輸出資料夾>` 只留 Trash 並改成 class 0。
  `--src` 底下要是每支影片一個 CVAT「YOLO 1.1」匯出（`<影片>/obj_train_data/*.txt`）；
  它只輸出標註檔，影像要自己搬成 `images/<split>/<影片>/` 與 `labels/<split>/<影片>/`。

---

## 2. 資料集

### 2.1 版本演變

所有影像都來自同一批監視器影片加上一份公開資料，版本之間的差別在「背景幀留多少」、
「有沒有加公開資料」，以及實驗 A 把 train 裡的監視器畫面整個拿掉。

現在能拿來訓練的有兩份：**★ 正式版**（有背景幀，目前所有模型比較都用它）與
**☆ labeled_only**（只有有垃圾的幀，比較小、跑得快）。

```
原始資料（NAS /mnt/backups/weihong/）
  datasetA   Google Drive 匯出的 50 個 zip（002、003 遺失）  54,711 張 PNG、22,667 個標註檔
  八德/       2 個 zip，補上遺失分卷裡那兩個資料夾的影像      7,284 張 PNG，沒有標註
  datasetB   Roboflow 公開資料 2 個 zip                     12,081 張 JPG、24,330 框
       │
       ▼  datasetA + 八德；沒有標註檔的影像補一個空 .txt（= 背景幀）
final_dataset                          61,995 幀｜25,123 框｜背景 46,185（74.5%）
       │
       ├─▶ 每個資料夾最多留「有垃圾幀數」那麼多張背景
       │   final_dataset_balanced_1to1       26,207 幀｜25,123 框｜背景 10,397
       │        │
       │        ▼  + datasetB（變成 opendataset_1、opendataset_2 兩個資料夾）
       │   ★ balanced_1to1_plus_opendataset  38,288 幀｜49,453 框｜背景 11,029（28.8%）
       │        │
       │        └─▶ 實驗 A：train 只放公開資料（9,110 張），test = 下面的評估子集（6,093 幀）
       │              expA_openTrainVal_nonOpenTest       18,068 幀（val 也只放公開資料；v7–v11 用）
       │              balanced_openTrainVal_nonOpenTest   22,549 幀（val = 正式版整個 validation；
       │                                                  v12–v13 用，symlink 已全斷）
       │
       └─▶ 丟掉所有背景幀
           ☆ final_dataset_labeled_only      15,810 幀｜25,123 框｜背景 0
                ├─▶ JPEG 版（q95）           同上，6.0 GB
                └─▶ 5 GB 小資料包             3,239 幀｜5,133 框｜背景 0
```

| 版本 | 大小 | 位置 | 用在 |
|---|---:|---|---|
| final_dataset | 110 GB | NAS | 只是中間產物 |
| final_dataset_balanced_1to1 | 42 GB | NAS | 只是中間產物 |
| **★ balanced_1to1_plus_opendataset** | 44 GB | `datasets/` | **v2–v6、v14–v19，以及所有 test 評估** |
| expA_openTrainVal_nonOpenTest | 13 GB | `datasets/` | v7–v11 |
| balanced_openTrainVal_nonOpenTest | symlink | `datasets/` | v12–v13（連結全斷，可修，見 `KNOWN_ISSUES.md` ISSUE-05） |
| **☆ final_dataset_labeled_only** | 25 GB | `datasets/` | **只有正樣本，改架構時快速迭代用**（2026-08-17 建立，還沒有正式訓練過） |
| labeled_only JPEG 版 | 6.0 GB | NAS | 方便傳輸 |
| 小資料包 | 5.1 GB | `handoff_smoke_20260820/` | 跑通流程 |

NAS 上也有一份正式版，影像與標註相同，但它 `data.yaml` 的 `path:` 還指向已刪除的舊路徑，要用得先改。

格式全部是 YOLO：`nc: 1`、`names: {0: trash}`，**空的 `.txt` = 這幀沒有垃圾（背景幀）**。

### 2.2 兩份訓練用資料的切分

兩份都依**影片資料夾**切，同一支影片只會出現在一個 split。

**★ 正式版（balanced_1to1_plus_opendataset）**

| split | 資料夾 | 幀 | 框 | 背景幀 |
|---|---:|---:|---:|---:|
| train | 36（監視器 34 + 公開 2） | 24,743 | 36,277 | 6,420（25.9%） |
| validation | 11（監視器 9 + 公開 2） | 7,346 | 9,451 | 1,748（23.8%） |
| test | 18（監視器） | 6,093 | 3,327 | 2,859（46.9%） |

所有 test 數字都是在這 18 支影片、6,093 幀（有垃圾 3,234 / 背景 2,859）上量的。
硬碟上的 test 資料夾另外還有 2 個公開資料夾（106 幀、398 框），評估時一律不用，所以不列。

**☆ labeled_only**（= 正式版去掉公開資料、去掉所有背景幀，逐檔比對過）

| split | 資料夾 | 幀 | 框 | 背景幀 |
|---|---:|---:|---:|---:|
| train | 34 | 9,730 | 18,278 | 0 |
| validation | 9 | 2,846 | 3,518 | 0 |
| test | 18 | 3,234 | 3,327 | 0 |

### 2.3 標註類別

| | 類別 |
|---|---|
| 原始標註（CVAT） | 3 個物件（Person / Vehicle / Trash）加上動作／狀態屬性，展開就是 8 類（person、person_holding、person_littering、vehicle、vehicle_holding、vehicle_littering、trash、trash_flying）。原始檔已不在本機與 NAS |
| 上游 datasetA 給的 | 只留垃圾，分 2 類：0 = 地上的垃圾（24,274 框）、1 = 飛行中的垃圾（849 框） |
| 正式版（2026-04-14 建立時） | 合併成 1 類 `trash`，所有模型都用單類訓練 |
| 之後的新標註 | 一樣從 CVAT 匯出 3 類，訓練前只留 Trash（見第 1 節） |

這些是同一份標註的不同展開，不是先後不同的資料。公開資料在上游就已經全部是 class 0。

---

## 3. 訓練

### 3.1 比較過的五個版本

五版都用正式版資料、從 COCO 預訓練權重開始、imgsz 1280。

| 版本 | 設定 | 訓練了幾個 epoch |
|---|---|---|
| **v4** | mosaic 開、早停（patience 20）、100 epoch、batch 8 | 第 72 epoch 早停 |
| v14 | mosaic 關、不早停、400 epoch、batch 10 | 跑滿 400 |
| v16 | mosaic 關、不早停、600 epoch、batch 10 | 跑滿 600 |
| v17 | v16 + 改 loss 權重（box/cls/dfl 7.5/0.5/1.5 → 3.0/1.5/0.5） | 跑滿 600 |
| v19 | v16 + mosaic 開、早停（patience 100） | 第 253 epoch 早停 |

v4 是在 RTX 4090（Ultralytics 8.4.37）上訓練的，其餘四版在 RTX 5090（8.4.41）上。

### 3.2 怎麼訓練

```bash
cd /home/weihong
python3 training/train_starter.py --check-only                 # 檢查資料集（不訓練、不寫檔）
nohup python3 -u training/train_starter.py --name <自己取名> > nohup_logs/<自己取名>.log 2>&1 &
```

`train_starter.py` 的預設值就是 v4 配方（100 epoch、imgsz 1280、batch 8、patience 20、mosaic 開、cos_lr）。
輸出在 `runs/detect/starter/<名字>/`：`weights/best.pt`、`weights/last.pt`、`results.csv`。
**名字要自己取**，同名會覆寫。

要重現某一版，用 `handoff_20260820_training/records/v*/train_v*.py`（精簡版，輸出名稱會加 `_repro`）。
**不要跑 `training/` 底下的原始實驗腳本**（`train_v*_600epoch.py`、`train.py`、`train_v14_400epoch_recovered.py`）：
它們綁了學長的 W&B 帳號，而且沿用原始 run 名稱，一跑就會覆寫 `runs/detect/` 裡的原始紀錄。

`best.pt` 是 val mAP50-95 最高的那一輪，`last.pt` 是最後一輪，兩個都要測。

---

## 4. 推論與測試

### 4.1 推論：產生偵測結果 CSV

```bash
cd /home/weihong
python3 relation_module/run_v16_detect_only.py \
  --weights handoff_20260820_training/weights/v4_last.pt \
  --source-root datasets/balanced_1to1_plus_opendataset/images/test \
  --output-csv relation_module/inputs_v4/<新檔名>.csv --conf 0.25
```

- 設定：conf 0.25、imgsz 1280（自動沿用訓練時的值）、NMS IoU 0.7。
- 輸出每個框一列：`資料夾路徑, 圖片檔名, 類別ID, 信心度, 左上X, 左上Y, 右下X, 右下Y`。沒有框的幀不會出現。
- **`--source-root` 和 `--output-csv` 一定要給。** 前者預設是已刪除的路徑；後者預設會覆寫評估腳本用來比對的參考檔。

### 4.2 測試：box-level

每個預測框都要跟標註框比位置，重疊 IoU ≥ 0.5 才算框對；沒對到標註物的框算誤報，沒被框到的標註算漏抓。

| 指標 | 怎麼算 |
|---|---|
| **mAP50** | Ultralytics val（conf 0.001，掃過所有信心度門檻），IoU ≥ 0.5 算對 |
| **mAP50-95** | 同上，IoU 0.5–0.95 十個門檻的平均，要求更嚴 |
| **TP（框對）** | 模型的框對到一個標註框（IoU ≥ 0.5）。conf 0.25 的框依信心度由高到低一對一配對 |
| **FP（誤報）** | 模型的框沒對到任何標註框 |
| **FN（漏抓）** | 標註框沒被任何模型的框對到 |
| **TN** | box-level **沒有 TN**：「沒垃圾、模型也沒畫框」的地方沒辦法一個一個數 |

### 4.3 結果

同一批 6,093 幀、3,327 個標註框。每版取 best.pt / last.pt 裡 mAP50 較高的那個。

| 模型 | mAP50 | mAP50-95 | TP | FP | FN | TN | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v4（best.pt） | **7.4** | 2.9 | **525** | 2,179 | 2,802 | — | 19.4% | **15.8%** | **17.4** |
| v19（last.pt） | 7.1 | **3.2** | 155 | 1,660 | 3,172 | — | 8.5% | 4.7% | 6.0 |
| v14（last.pt） | 1.5 | 0.9 | 67 | 256 | 3,260 | — | **20.7%** | 2.0% | 3.7 |
| v16（best.pt） | 1.4 | 1.0 | 89 | 3,524 | 3,238 | — | 2.5% | 2.7% | 2.6 |
| v17（last.pt） | 0.7 | 0.2 | 100 | 1,476 | 3,227 | — | 6.3% | 3.0% | 4.1 |

算式（TP + FN 固定是 3,327，也就是標註框的總數）：

```
precision = TP / (TP + FP)          畫出來的框裡，有幾成是對的。低 = 亂報警多
recall    = TP / (TP + FN)          標註框裡，抓到幾成。低 = 漏抓多
F1        = 2 × precision × recall / (precision + recall)

例：v4（best.pt）
precision = 525 / (525 + 2,179) = 19.4%
recall    = 525 / (525 + 2,802) = 15.8%
F1        = 2 × 19.4 × 15.8 / (19.4 + 15.8) = 17.4
```

- TP / FP / FN / precision / recall / F1 用 conf 0.25；mAP 兩欄是 Ultralytics val（conf 0.001）。
- **v4 最會抓**：recall 15.8%，是其他版本的 3 倍以上；但它畫的 2,704 個框裡有 2,179 個是錯的（precision 19.4%），每 5 個框約 4 個是誤報。
- 其他版本不是比較少亂報，而是抓得更少：v14 的 precision 最高（20.7%），但只抓到 2%；v16 畫了最多框（3,613 個），precision 只有 2.5%。
- v19 的 mAP50-95 最高，代表它框對的那些位置比較準，但數量少（recall 4.7%）。
- 所有版本的 mAP50 都低於 8%。原因之一是標註只標「被丟出來的那一件」，場景裡原本就有的垃圾不標，
  模型框到那些會被算成誤報。

### 4.4 folder-level：18 支 test 影片抓到幾支

以整支影片為單位：一支影片裡只要有一幀「模型報警、而且那幀真的有垃圾」就算抓到（不看框的位置）。

- **報警** = 這一幀至少有一個框的信心度 ≥ **0.25**（模型每畫一個框都會給一個 0–1 的信心度，低於 0.25 的框直接丟掉）。
- **真的有垃圾** = 這一幀的標註檔不是空的。

每版用的 checkpoint 與 4.3 相同。

| 模型 | 抓到（共 18 支） |
|---|---:|
| v4（best.pt） | **11** |
| v19（last.pt） | 10 |
| v16（best.pt） | 8 |
| v17（last.pt） | 8 |
| v14（last.pt） | 7 |

---

## 5. 東西在哪

| 位置 | 內容 |
|---|---|
| `handoff_smoke_20260820/` | 5 GB 小資料包 + `train.py` |
| `handoff_20260820_training/` | **主要交接包**：五版權重（v4/v14/v16/v17/v19 的 best 與 last）、訓練設定、評估輸出、`KNOWN_ISSUES.md` |
| `handoff_20260810/` | 完整研究紀錄：SAHI、第二階段、v6/v11/v13/v15/v18 |
| `datasets/` | 正式版（44 GB）與 labeled_only（25 GB），**不在交接包裡** |
| `/mnt/backups/weihong/`（NAS） | 原始 zip、final_dataset、balanced_1to1、正式版的副本 |
| `results/v4_last_top10_per_video_20261001/` | v4 last 的標註 vs 預測對照圖（10 支影片，每支 10 張） |
| `CLAUDE.md` | 每支腳本怎麼跑、有哪些坑 |

動任何腳本前先查 `handoff_20260820_training/KNOWN_ISSUES.md`。
