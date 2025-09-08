from ultralytics import YOLO
from pathlib import Path
import sys

DATA_CFG = "configs/data.yaml"
TRAIN_CFG = "configs/train.yaml"

def main():
    dataset_root = Path("data/yolo")

    model = YOLO("yolov8n.pt")
    model.train(
        data=DATA_CFG,
        cfg=TRAIN_CFG,
        project="runs/train",
        name="baseline"
    )
    model.val(
        data=DATA_CFG,
        imgsz=640,
        conf=0.001,
        iou=0.5,
        plots=True
    )

if __name__ == "__main__":
    main()