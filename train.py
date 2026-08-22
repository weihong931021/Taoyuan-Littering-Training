"""最小可執行的 YOLO11 訓練腳本。

    python3 train.py                      # 用預設設定訓練
    python3 train.py --epochs 1 --imgsz 640 --name smoke    # 先跑通流程
    python3 train.py --device cpu         # 沒有 GPU

輸出在 runs/detect/trash_detect/<name>/：weights/best.pt、weights/last.pt、results.csv
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=HERE / "dataset/data.yaml")
    p.add_argument("--model", default=HERE / "yolo11l.pt")
    p.add_argument("--name", default="run")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    a = p.parse_args()

    # ★ 這一行就是「使用預訓練」的地方。
    #   YOLO("yolo11l.pt")   -> 載入 COCO 預訓練權重（我們要的）
    #   YOLO("yolo11l.yaml") -> 隨機初始化，從零開始練
    #
    #   注意：model.train() 裡的 pretrained=True/False 在偵測訓練中是**無效參數**，
    #   設了也不會有任何作用。決定權完全在這一行。
    model = YOLO(a.model)

    model.train(
        data=a.data,
        name=a.name,
        project="trash_detect",
        epochs=a.epochs,
        imgsz=a.imgsz,      # 專案標準值 1280，所有圖會 letterbox 到這個尺寸
        batch=a.batch,      # VRAM 不夠就調小
        device=a.device,
        # patience：連續幾個 epoch 沒進步就早停。預設是 100。
        #   v4 用 20（較嚴格）
        #   v6~v18 全部設 0（= 永不早停，硬跑滿）
        patience=20,
        # mosaic 刻意不寫：Ultralytics 預設就是開的（1.0）。
        #   v6~v18 曾主動寫 mosaic=0.0 關掉它，結果變差。不要加這個參數。
        cos_lr=True,        # 學習率照餘弦曲線衰減
        workers=6,
        seed=0,             # 固定亂數種子，實驗才可重現
    )


if __name__ == "__main__":
    main()
