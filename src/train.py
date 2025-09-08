from ultralytics import YOLO
from pathlib import Path
import sys

import mlflow, dagshub
dagshub.init(repo_owner="KraviGO", repo_name="tbank-logo-detector", mlflow=True)
mlflow.set_experiment("tbank_baseline")

DATA_CFG = "configs/data.yaml"
TRAIN_CFG = "configs/train.yaml"

def main():
    dataset_root = Path("data/yolo")

    with mlflow.start_run(run_name="yolov8n_640_e25"):
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

        mlflow.log_param("model", "yolov8n.pt")
        mlflow.log_param("imgsz", 640)

        best = Path("runs/train/baseline/weights/best.pt")
        if best.exists():
            mlflow.log_artifact(str(best), artifact_path="weights")
        pr = Path("runs/train/baseline/PR_curve.png")
        if pr.exists():
            mlflow.log_artifact(str(pr))

if __name__ == "__main__":
    main()