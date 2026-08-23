# 標註版本說明（桃園／新屋 17 部影片）

所有版本都是從 CVAT 匯出的原始 YOLO 標註衍生，**每個版本都是 17 部影片、4,634 個 txt，座標完全相同**，只差 class id 與有沒有過濾。已逐檔比對確認一致。

影格固定 1280×720，檔名 `frame_XXXXXX.txt`，YOLO 格式 `class cx cy w h`（正規化）。

## 版本對照

| 版本 | class 定義 | box 數 | 產生方式 |
|---|---|---:|---|
| **origin_yolo**（原始） | `0 Person` / `1 Vehicle` / `2 Trash`（nc=3） | 13,582 | CVAT 匯出，附 PNG 影格 |
| **labels_8class** | `0 person` `1 person_holding` `2 person_littering` `3 vehicle` `4 vehicle_holding` `5 vehicle_littering` `6 trash` `7 trash_flying` | 13,582 | `split_8class_labels.py`：用 CVAT XML 的 Action / State 屬性展開 |
| **labels_split** | `0 Trash` / `1 Trash_Flying` | 9,723 | `split_trash_labels.py`：只留 Trash，依 XML State 拆 |
| **filtered_labels** | `0 trash`（nc=1） | 9,723 | `filter_trash_labels.py`：只留 Trash，class 2 → 0 |

8 class box 分布：person 521 / person_holding 742 / person_littering 22 / vehicle 2,114 / vehicle_holding 340 / vehicle_littering 120 / trash 9,663 / trash_flying 60。

## filter_trash_labels.py 用法

```bash
python filter_trash_labels.py                                  # 預設 origin_yolo → filtered_labels
python filter_trash_labels.py --src <原始資料夾> --out <輸出資料夾>
```

`--src` 底下要是 `<影片>/obj_train_data/*.txt` 的結構。沒有 Trash 的影格會寫出空檔（負樣本）。

## 各影片統計

| 影片資料夾 | 影格 | 原始 box | Trash | Flying | 無 Trash 影格 |
|---|---:|---:|---:|---:|---:|
| 新屋_327-04-62_20250604103457_processed_1 | 61 | 159 | 98 | 0 | 0 |
| 新屋_327-04-62_20250604103457_processed_2 | 18 | 34 | 10 | 6 | 2 |
| 新屋_327-04-62_20250613084550_processed | 52 | 62 | 3 | 7 | 42 |
| 桃園_PIR_0140_processed | 84 | 168 | 74 | 10 | 0 |
| 桃園_PIR_2460_processed | 147 | 213 | 61 | 5 | 81 |
| 桃園_PIR_2981_processed | 145 | 154 | 5 | 4 | 136 |
| 桃園_PIR_3004_processed | 140 | 157 | 9 | 8 | 123 |
| 桃園_PIR_3016_processed | 66 | 93 | 23 | 4 | 39 |
| 桃園_PIR_3150_processed | 263 | 325 | 55 | 7 | 201 |
| 桃園_PIR_3245_processed | 352 | 373 | 17 | 4 | 331 |
| 桃園_PIR_4175_processed | 270 | 332 | 58 | 4 | 208 |
| 桃園_TLC00005_processed | 8 | 15 | 7 | 0 | 1 |
| 桃園_TLC00111 6-19_processed | 1380 | 3804 | 3685 | 0 | 220 |
| 桃園_TLC00147 6-19_processed | 207 | 213 | 6 | 0 | 201 |
| 桃園_TLC00159 6-19_processed | 25 | 40 | 23 | 0 | 2 |
| 桃園_TLC00161 6-19_processed | 713 | 6250 | 5043 | 0 | 1 |
| 桃園影片_2025_11月17日_1280x720 | 703 | 1190 | 486 | 1 | 216 |
| **總計** | **4,634** | **13,582** | **9,663** | **60** | **1,804** |

## 注意

- 資料夾名稱含空格（`桃園_TLC00111 6-19_processed` 等），shell 處理要 quote。
- Trash_Flying 只有 60 個、person_littering 只有 22 個，類別極度不平衡。
- TLC00111 / TLC00161 兩部佔了九成以上的 Trash box。
