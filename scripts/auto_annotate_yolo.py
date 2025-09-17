from ultralytics import YOLO
from pathlib import Path
from typing import List, Iterable, Optional
import shutil
import os

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **k: x  # если tqdm не установлен

def list_images(root: Path, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]

def chunks(seq: List[Path], size: int) -> Iterable[List[Path]]:
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def autosel_device(prefer: Optional[str] = None) -> str:
    """
    Возвращает "cuda"/"mps"/"cpu".
    prefer: можно явно задать "cuda"/"mps"/"cpu" — тогда вернёт его при доступности.
    """
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

def xyxy_to_xywhn(xyxy, orig_w: int, orig_h: int):
    """
    Конвертирует [x1, y1, x2, y2] -> нормированный (x, y, w, h), центр + ширина/высота.
    """
    x1, y1, x2, y2 = map(float, xyxy)
    w = x2 - x1
    h = y2 - y1
    xc = x1 + w / 2.0
    yc = y1 + h / 2.0
    # нормализация
    return (
        max(0.0, min(1.0, xc / orig_w)),
        max(0.0, min(1.0, yc / orig_h)),
        max(0.0, min(1.0, w / orig_w)),
        max(0.0, min(1.0, h / orig_h)),
    )

def write_yolo_label(label_path: Path, rows: List[str]):
    ensure_dir(label_path.parent)
    with open(label_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(r + "\n")

def main(
    weights: str = "runs/train/baseline/weights/best.pt",
    dataset_images_root: str = "data/yolo/images",  # можно указать конкретно train/val и т.п.
    out_root: str = "autolabeled_yolo",
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.7,
    batch: int = 16,
    device: Optional[str] = None,       # "cuda" / "mps" / "cpu" или None для авто
    save_annotated: bool = False,       # рисованные предпросмотры
    write_empty_labels: bool = False,   # писать пустые .txt для кадров без детектов
    class_filter: Optional[List[int]] = None,  # например [0] — оставляем только класс 0
):
    """
    Создаёт структуру:
      out_root/
        images/relative/path/to/img.jpg
        labels/relative/path/to/img.txt
        classes.txt
        data.yaml   # готово для Ultralytics/Roboflow

    По умолчанию копируем ТОЛЬКО изображения с детектами. Если write_empty_labels=True,
    то будут созданы и пустые labels для негативов (YOLO допускает пустой файл).
    """
    dataset_images_root = Path(dataset_images_root)
    out_root = Path(out_root)
    out_images_root = out_root / "images"
    out_labels_root = out_root / "labels"
    ensure_dir(out_images_root)
    ensure_dir(out_labels_root)

    # устройство
    dev = autosel_device(device)
    print(f"[INFO] Using device: {dev}")

    # модель
    model = YOLO(weights)
    class_names = model.names if hasattr(model, "names") else None
    if isinstance(class_names, dict):
        # Ultralytics часто даёт словарь {id: name}
        class_names = [class_names[i] for i in sorted(class_names.keys())]
    if not class_names:
        # запасной вариант
        class_names = ["logo"]

    # список картинок
    all_imgs = list_images(dataset_images_root)
    if not all_imgs:
        print(f"[WARN] No images found in {dataset_images_root.resolve()}")
        return

    print(f"[INFO] Found {len(all_imgs)} images. Running inference & writing YOLO labels...")

    # подготовим classes.txt — удобно для Roboflow
    with open(out_root / "classes.txt", "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(str(name) + "\n")

    kept = 0
    empties = 0

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
            # размеры оригинала известны из результата
            H, W = res.orig_shape  # (h, w)

            rows = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy().astype(int)
                confs = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else None

                for i in range(len(xyxy)):
                    cid = int(cls[i])
                    if class_filter is not None and cid not in class_filter:
                        continue

                    x, y, w, h = xyxy_to_xywhn(xyxy[i], W, H)
                    # Формат строки: class x y w h [можно без conf — классический YOLO]
                    rows.append(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

            rel = img_path.relative_to(dataset_images_root)
            out_img = out_images_root / rel
            out_lbl = out_labels_root / rel.with_suffix(".txt")
            ensure_dir(out_img.parent)
            ensure_dir(out_lbl.parent)

            if rows:
                # есть детекты → копируем картинку и пишем label
                shutil.copy2(img_path, out_img)
                write_yolo_label(out_lbl, rows)
                kept += 1
            else:
                if write_empty_labels:
                    # создаём пустой .txt и копируем изображение (полезно для последующего доразмечивания)
                    shutil.copy2(img_path, out_img)
                    write_yolo_label(out_lbl, [])
                    empties += 1
                # иначе пропускаем “пустые” кадры полностью

            # предпросмотры (опционально)
            if rows and save_annotated:
                try:
                    import cv2
                    ann = res.plot()  # bgr ndarray
                    ann_path = (out_root / "annotated" / rel).with_suffix(".jpg")
                    ensure_dir(ann_path.parent)
                    cv2.imwrite(str(ann_path), ann)
                except Exception as e:
                    print(f"[WARN] Failed to save preview for {img_path}: {e}")

    # data.yaml для Ultralytics (images->train/val можно задать вручную; тут — на весь набор)
    data_yaml = out_root / "data.yaml"
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(
            "path: " + str(out_root.resolve()) + "\n"
            "train: images\n"
            "val: images\n"
            "test: \n"
            f"nc: {len(class_names)}\n"
            "names:\n" + "".join([f"  - {n}\n" for n in class_names])
        )

    print(f"[DONE] Images with labels: {kept}")
    if write_empty_labels:
        print(f"[INFO] Empty labels created: {empties}")
    print(f"[OUT] Dataset root: {out_root.resolve()}")
    print(f"[OUT] Images: {out_images_root.resolve()}")
    print(f"[OUT] Labels: {out_labels_root.resolve()}")
    print(f"[OUT] classes.txt: {(out_root/'classes.txt').resolve()}")
    print(f"[OUT] data.yaml: {data_yaml.resolve()}")

if __name__ == "__main__":
    main(
        weights="models/yolov8n_epochs_25.pt",
        dataset_images_root="filtered_with_detections/images",
        out_root="autolabeled_yolo",
        imgsz=640,
        conf=0.25,
        iou=0.7,
        batch=16,
        device="mps",
        save_annotated=False,
        write_empty_labels=False,
        class_filter=None
    )