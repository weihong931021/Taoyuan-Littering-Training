#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
從原始 3 class YOLO 標註只保留 Trash，輸出單類標註
- 來源: origin_yolo/<影片>/obj_train_data/*.txt  (0 Person / 1 Vehicle / 2 Trash)
- 輸出: filtered_labels/<影片>/*.txt              (0 trash)
- 座標原樣保留，只改 class id；沒有 Trash 的影格會寫出空檔

原本是 train.py 裡的 filter_trash_labels()，抽出來獨立執行。

用法:
    python filter_trash_labels.py
    python filter_trash_labels.py --src origin_yolo --out filtered_labels
"""

import argparse
from pathlib import Path

# ===================== 設定 =====================
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / 'origin_yolo'
OUTPUT_DIR = BASE_DIR / 'filtered_labels'

TRASH_CLASS_ID = '2'   # 原始 class
NEW_CLASS_ID = '0'     # 輸出 class
# ================================================


def find_video_folders(src_dir: Path):
    """找出所有含 obj_train_data 的影片資料夾"""
    return sorted(p for p in src_dir.iterdir()
                  if p.is_dir() and (p / 'obj_train_data').is_dir())


def filter_one_video(video_dir: Path, out_dir: Path):
    """回傳 (總影格數, 含 Trash 影格數, Trash box 數)"""
    src = video_dir / 'obj_train_data'
    out_dir.mkdir(parents=True, exist_ok=True)

    total_frames = trash_frames = trash_boxes = 0

    for txt in sorted(src.glob('*.txt')):
        total_frames += 1
        kept = []
        for line in txt.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0] == TRASH_CLASS_ID:
                kept.append(f"{NEW_CLASS_ID} {' '.join(parts[1:5])}\n")

        (out_dir / txt.name).write_text(''.join(kept))

        if kept:
            trash_frames += 1
            trash_boxes += len(kept)

    return total_frames, trash_frames, trash_boxes


def main():
    ap = argparse.ArgumentParser(description='只保留 Trash 的 YOLO 標註過濾')
    ap.add_argument('--src', type=Path, default=SRC_DIR, help='origin_yolo 路徑')
    ap.add_argument('--out', type=Path, default=OUTPUT_DIR, help='輸出路徑')
    args = ap.parse_args()

    videos = find_video_folders(args.src)
    if not videos:
        print(f"⚠️ 在 {args.src} 找不到任何含 obj_train_data 的資料夾")
        return

    print(f"🔍 過濾 Trash 類別 ({TRASH_CLASS_ID} → {NEW_CLASS_ID})")
    print(f"   來源: {args.src}")
    print(f"   輸出: {args.out}\n")
    print(f"{'影片資料夾':<48}{'總影格':>8}{'含Trash':>9}{'Trash box':>11}{'比例':>8}")
    print('-' * 84)

    g_frames = g_trash_frames = g_boxes = 0
    for v in videos:
        frames, tf, boxes = filter_one_video(v, args.out / v.name)
        g_frames += frames; g_trash_frames += tf; g_boxes += boxes
        ratio = tf / frames if frames else 0
        print(f"{v.name:<48}{frames:>8}{tf:>9}{boxes:>11}{ratio:>8.1%}")

    print('-' * 84)
    print(f"{'總計':<48}{g_frames:>8}{g_trash_frames:>9}{g_boxes:>11}{g_trash_frames / g_frames:>8.1%}")
    print(f"\n✅ 完成！{len(videos)} 部影片，輸出位置: {args.out}")


if __name__ == '__main__':
    main()
