#!/usr/bin/env python3
"""
從 final_dataset_labeled_only（25G，全部都是有標註的正樣本）等比例抽出一份小的
子集，供他人跑通整條 training pipeline。

抽樣規則
--------
1. **保留全部 61 個資料夾與 train/validation/test 三個 split**，只在每個資料夾內部抽幀。
   （不是丟掉整個資料夾 —— 那會讓場景多樣性直接消失。）
2. 資料夾內用**均勻時間間隔**抽幀（不是取前 N 張），因為同一資料夾是一段連續影片，
   取前 N 張等於只拿到影片開頭幾秒。
3. 每個資料夾至少留 `--min-frames` 張（預設 3），避免小資料夾被抽成 0 張。
4. 抽樣比例由**目標容量**反推：對 ratio 做二分搜尋，用實際檔案大小試算，
   直到總量落在 `--target-gib` 附近。因為各資料夾的檔案大小不一，
   單純用「幀數 20%」不會剛好等於「容量 20%」。

輸出的子集結構與來源相同（images/ 與 labels/ 兩棵鏡像樹），可直接餵給 Ultralytics。

用法
----
    python3 build_subset.py --dry-run          # 只試算，不複製
    python3 build_subset.py                    # 實際產生 dataset/
    python3 build_subset.py --target-gib 2.0   # 想要更小的話
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

SRC_DEFAULT = Path("/home/weihong/datasets/final_dataset_labeled_only")
SPLITS = ["train", "validation", "test"]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

DATA_YAML = """\
# 由 build_subset.py 產生的小型子集（來源：final_dataset_labeled_only）
# 這份資料**只有正樣本**（每張圖都有標註），沒有背景幀。
# 所以在這份 test 上量到的 precision / F1 會虛高 —— 它量不到誤報。
# 它的用途是「跑通 pipeline」，不是「產生可對外報告的分數」。
path: {path}
train: images/train
val: images/validation
test: images/test

nc: 1
names:
  0: trash
"""


def scan(src: Path):
    """回傳 {split: [(folder_name, [(image_path, bytes), ...]), ...]}"""
    tree = {}
    for split in SPLITS:
        split_dir = src / "images" / split
        if not split_dir.is_dir():
            sys.exit(f"[fatal] 找不到 {split_dir}")
        folders = []
        for folder in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            frames = sorted(
                (f, f.stat().st_size)
                for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES
            )
            if frames:
                folders.append((folder.name, frames))
        tree[split] = folders
    return tree


def pick(frames, keep):
    """在 len(frames) 張中均勻取 keep 張，涵蓋整段影片的頭尾。"""
    n = len(frames)
    if keep >= n:
        return list(range(n))
    return [round(i * (n - 1) / (keep - 1)) if keep > 1 else 0 for i in range(keep)]


def plan(tree, ratio, min_frames):
    """依 ratio 選出要保留的幀，回傳 (selection, total_bytes, total_frames)。"""
    selection, total_bytes, total_frames = {}, 0, 0
    for split, folders in tree.items():
        picked_folders = []
        for name, frames in folders:
            keep = max(min(len(frames), min_frames), round(len(frames) * ratio))
            idx = pick(frames, keep)
            chosen = [frames[i] for i in idx]
            total_bytes += sum(b for _, b in chosen)
            total_frames += len(chosen)
            picked_folders.append((name, len(frames), [p for p, _ in chosen]))
        selection[split] = picked_folders
    return selection, total_bytes, total_frames


def solve_ratio(tree, target_bytes, min_frames):
    """二分搜尋出讓總容量最接近 target_bytes 的 ratio。"""
    lo, hi = 0.0, 1.0
    best = None
    for _ in range(40):
        mid = (lo + hi) / 2
        selection, size, frames = plan(tree, mid, min_frames)
        if best is None or abs(size - target_bytes) < abs(best[1] - target_bytes):
            best = (selection, size, frames, mid)
        if size > target_bytes:
            hi = mid
        else:
            lo = mid
    return best


def main() -> int:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="等比例抽出小型訓練子集")
    p.add_argument("--src", type=Path, default=SRC_DEFAULT)
    p.add_argument("--dst", type=Path, default=here / "dataset")
    p.add_argument("--target-gib", type=float, default=5.0,
                   help="目標容量（GiB，與 du -sh 的顯示一致）。預設 5.0")
    p.add_argument("--min-frames", type=int, default=3,
                   help="每個資料夾至少保留幾張。預設 3")
    p.add_argument("--dry-run", action="store_true", help="只試算，不複製檔案")
    a = p.parse_args()

    print(f"[scan] 掃描來源 {a.src}")
    tree = scan(a.src)
    src_frames = sum(len(fr) for folders in tree.values() for _, fr in folders)
    src_bytes = sum(b for folders in tree.values() for _, fr in folders for _, b in fr)
    print(f"[scan] 來源：{src_frames} 幀 / {src_bytes / 2**30:.2f} GiB / "
          f"{sum(len(f) for f in tree.values())} 個資料夾")

    target = a.target_gib * 2**30
    selection, size, frames, ratio = solve_ratio(tree, target, a.min_frames)
    print(f"[plan] 抽樣比例 {ratio * 100:.2f}%  ->  {frames} 幀 / {size / 2**30:.2f} GiB "
          f"(目標 {a.target_gib} GiB)")
    for split in SPLITS:
        kept = sum(len(f[2]) for f in selection[split])
        orig = sum(len(fr) for _, fr in tree[split])
        print(f"       {split:11s} {kept:5d} / {orig:5d} 幀 "
              f"({kept / orig * 100:.1f}%)  {len(selection[split])} 個資料夾")

    if a.dry_run:
        print("[dry-run] 未複製任何檔案")
        return 0

    if a.dst.exists():
        sys.exit(f"[fatal] {a.dst} 已存在。請先移除或改用 --dst")

    manifest_rows = []
    copied = missing_labels = 0
    for split in SPLITS:
        for folder_name, orig_count, images in selection[split]:
            img_dir = a.dst / "images" / split / folder_name
            lbl_dir = a.dst / "labels" / split / folder_name
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            for img in images:
                shutil.copy2(img, img_dir / img.name)
                label = a.src / "labels" / split / folder_name / f"{img.stem}.txt"
                if label.exists():
                    shutil.copy2(label, lbl_dir / label.name)
                else:
                    missing_labels += 1
                copied += 1
                manifest_rows.append({
                    "split": split,
                    "folder": folder_name,
                    "image": img.name,
                    "folder_frames_in_source": orig_count,
                })
            print(f"[copy] {split}/{folder_name}: {len(images)} / {orig_count}")

    (a.dst / "data.yaml").write_text(
        DATA_YAML.format(path=a.dst.resolve()), encoding="utf-8")

    with open(here / "SUBSET_MANIFEST.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["split", "folder", "image", "folder_frames_in_source"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\n[done] 複製 {copied} 幀 -> {a.dst}")
    if missing_labels:
        print(f"[warn] 有 {missing_labels} 張圖找不到對應標記檔（來源不該發生，請回報）")
    print(f"[done] 清單寫入 {here / 'SUBSET_MANIFEST.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
