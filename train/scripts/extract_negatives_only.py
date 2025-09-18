# extract_negatives_only.py
from ultralytics import YOLO
from pathlib import Path
from typing import List, Iterable, Optional
import shutil

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **k: x

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def list_images(root: Path, exts=IMG_EXTS) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]

def chunks(seq: List[Path], size: int) -> Iterable[List[Path]]:
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def autosel_device(prefer: Optional[str] = None) -> str:
    """auto: cuda → mps → cpu"""
    try:
        import torch
    except Exception:
        return "cpu"
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    if prefer == "mps" and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def write_empty_label(label_path: Path):
    ensure_dir(label_path.parent)
    with open(label_path, "w", encoding="utf-8"):
        pass

def main(
    weights: str = "runs/train/baseline/weights/best.pt",
    dataset_images_root: str = "data/yolo/images",
    out_root: str = "negatives_only",
    imgsz: int = 640,
    conf: float = 0.5,
    iou: float = 0.7,
    batch: int = 16,
    device: Optional[str] = None,
    write_empty_labels: bool = True,
    min_box_rel_area: float = 0.0005  # отсекаем микробоксы
):
    dataset_images_root = Path(dataset_images_root)
    out_root = Path(out_root)
    out_images_root = out_root / "images"
    out_labels_root = out_root / "labels"

    ensure_dir(out_images_root)
    if write_empty_labels:
        ensure_dir(out_labels_root)

    dev = autosel_device(device)
    print(f"[INFO] Using device: {dev}")

    model = YOLO(weights)

    all_imgs = list_images(dataset_images_root)
    if not all_imgs:
        print(f"[WARN] No images in {dataset_images_root.resolve()}")
        return

    print(f"[INFO] Found {len(all_imgs)} images. Extracting negatives...")

    saved = 0
    for batch_paths in tqdm(list(chunks(all_imgs, batch)), desc="Batches"):
        results = model.predict(
            source=[str(p) for p in batch_paths],
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=dev,
            verbose=False
        )

        for img_path, res in zip(batch_paths, results):
            H, W = res.orig_shape
            has_logo = False

            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                for (x1, y1, x2, y2) in xyxy:
                    bw = max(0.0, x2 - x1)
                    bh = max(0.0, y2 - y1)
                    if (bw * bh) / (H * W) >= min_box_rel_area:
                        has_logo = True
                        break

            if not has_logo:
                rel = img_path.relative_to(dataset_images_root)
                dst_img = out_images_root / rel
                ensure_dir(dst_img.parent)
                shutil.copy2(img_path, dst_img)

                if write_empty_labels:
                    dst_lbl = out_labels_root / rel.with_suffix(".txt")
                    write_empty_label(dst_lbl)

                saved += 1

    print(f"[DONE] Negatives saved: {saved}")
    print(f"[OUT] Images: {out_images_root.resolve()}")
    if write_empty_labels:
        print(f"[OUT] Labels: {out_labels_root.resolve()}")

if __name__ == "__main__":
    main(
        weights="models/yolov8s_640_e50.pt",
        dataset_images_root="data/raw",
        out_root="negatives_only",
        imgsz=640,
        conf=0.5,
        iou=0.7,
        batch=16,
        device=None,
        write_empty_labels=True,
        min_box_rel_area=0.0005
    )