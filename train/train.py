from ultralytics import YOLO
from pathlib import Path
import mlflow, dagshub
import yaml

dagshub.init(repo_owner="KraviGO", repo_name="tbank-logo-detector", mlflow=True)
mlflow.set_experiment("tbank_baseline")

DATA_CFG = "configs/data.yaml"
TRAIN_CFG = "configs/train.yaml"

AUG_KEYS = [
    "hsv_h", "hsv_s", "hsv_v",
    "degrees", "translate", "scale", "shear", "perspective",
    "flipud", "fliplr",
    "mosaic", "mixup", "copy_paste", "erasing"
]

BASE_TRAIN_KEYS = [
    "imgsz", "epochs", "batch", "patience",
    "optimizer", "lr0", "lrf", "warmup_epochs", "weight_decay",
    "conf", "iou"
]

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def pick(d: dict, keys: list[str]) -> dict:
    return {k: d[k] for k in keys if k in d}

def main():
    cfg = load_yaml(TRAIN_CFG)
    aug_params = pick(cfg, AUG_KEYS)
    base_params = pick(cfg, BASE_TRAIN_KEYS)

    with mlflow.start_run(run_name="yolov8s_640_e50"):
        mlflow.log_artifact(TRAIN_CFG, artifact_path="configs")
        mlflow.log_artifact(DATA_CFG, artifact_path="configs")

        mlflow.log_param("model", "yolov8s.pt")
        for k, v in base_params.items():
            mlflow.log_param(k, v)

        mlflow.log_params({f"aug__{k}": v for k, v in aug_params.items()})

        model = YOLO("yolov8s.pt")
        model.train(
            data=DATA_CFG,
            cfg=TRAIN_CFG,
            project="runs/train",
            name="baseline"
        )

        model.val(
            data=DATA_CFG,
            imgsz=base_params.get("imgsz", 640),
            conf=base_params.get("conf", 0.25),
            iou=base_params.get("iou", 0.7),
            plots=True
        )

        trainer_args = getattr(getattr(model, "trainer", None), "args", None)
        if isinstance(trainer_args, dict):
            eff_aug = {k: trainer_args.get(k) for k in AUG_KEYS if k in trainer_args}
            eff_aug = {f"aug_effective__{k}": v for k, v in eff_aug.items()}
            if eff_aug:
                mlflow.log_params(eff_aug)

        run_dir = Path("runs/train/baseline")
        best = run_dir / "weights" / "best.pt"
        if best.exists():
            mlflow.log_artifact(str(best), artifact_path="weights")

        for img in [
            "PR_curve.png",
            "F1_curve.png",
            "P_curve.png",
            "R_curve.png",
            "results.png",
        ]:
            p = run_dir / img
            if p.exists():
                mlflow.log_artifact(str(p), artifact_path="plots")

if __name__ == "__main__":
    main()