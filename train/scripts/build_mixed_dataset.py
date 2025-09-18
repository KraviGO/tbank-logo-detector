from pathlib import Path
from typing import List, Tuple, Optional
import shutil
import random
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def list_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def non_empty_file(p: Path) -> bool:
    try:
        return p.is_file() and p.stat().st_size > 0
    except Exception:
        return False

def read_classes(root: Path) -> List[str]:
    for name in ["data.yaml", "dataset.yaml"]:
        y = root / name
        if y.exists():
            with open(y, "r", encoding="utf-8") as f:
                d = yaml.safe_load(f) or {}
            names = d.get("names")
            if isinstance(names, dict):
                return [names[i] for i in sorted(names.keys())]
            if isinstance(names, list) and names:
                return names
    txt = root / "classes.txt"
    if txt.exists():
        return [line.strip() for line in txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    return ["logo"]

def pair_from_pos(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    for img in list_images(images_dir):
        rel = img.relative_to(images_dir)
        lbl = labels_dir / rel.with_suffix(".txt")
        if non_empty_file(lbl):
            pairs.append((img, lbl))
    return pairs

def collect_positives(pos_root: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    i0, l0 = pos_root / "images", pos_root / "labels"
    if i0.exists() and l0.exists():
        pairs += pair_from_pos(i0, l0)
    for s in ["train", "valid", "test"]:
        i1, l1 = pos_root / "images" / s, pos_root / "labels" / s
        if i1.exists() and l1.exists():
            pairs += pair_from_pos(i1, l1)
        i2, l2 = pos_root / s / "images", pos_root / s / "labels"
        if i2.exists() and l2.exists():
            pairs += pair_from_pos(i2, l2)
    if not pairs:
        raise FileNotFoundError(f"Не нашёл позитивов в {pos_root}")
    return pairs

def collect_negatives(neg_root: Path) -> List[Tuple[Path, Path]]:
    imgs_dir = neg_root / "images"
    lbls_dir = neg_root / "labels"
    if not imgs_dir.exists() or not lbls_dir.exists():
        raise FileNotFoundError(f"В {neg_root} ожидаю images/ и labels/")
    pairs = []
    for img in list_images(imgs_dir):
        rel = img.relative_to(imgs_dir)
        lbl = lbls_dir / rel.with_suffix(".txt")
        pairs.append((img, lbl))  # пустой/не существующий .txt допустим — создадим
    return pairs

def split_counts(n: int) -> tuple[int, int, int]:
    tr = int(round(n * 0.7))
    va = int(round(n * 0.2))
    te = n - tr - va
    return tr, va, te

def stratified_mix(pos_list: List[Tuple[Path, Path]],
                   neg_list: List[Tuple[Path, Path]],
                   seed: int,
                   target_pos_ratio: float = 0.25):
    random.seed(seed)
    random.shuffle(pos_list)
    random.shuffle(neg_list)

    total_pos, total_neg = len(pos_list), len(neg_list)
    if total_pos == 0 or total_neg == 0:
        raise RuntimeError("Пустые позиции/негативы — проверь входные папки.")

    max_pos_by_neg = total_neg // 3
    use_pos = min(total_pos, max_pos_by_neg)
    use_neg = min(total_neg, use_pos * 3)
    if use_neg < use_pos * 3:
        use_pos = use_neg // 3

    pos_sel = pos_list[:use_pos]
    neg_sel = neg_list[:use_neg]

    pos_tr, pos_va, pos_te = split_counts(len(pos_sel))
    neg_tr, neg_va, neg_te = split_counts(len(neg_sel))

    pos_train = pos_sel[:pos_tr]
    pos_valid = pos_sel[pos_tr:pos_tr + pos_va]
    pos_test  = pos_sel[pos_tr + pos_va:]

    neg_train = neg_sel[:neg_tr]
    neg_valid = neg_sel[neg_tr:neg_tr + neg_va]
    neg_test  = neg_sel[neg_tr + neg_va:]

    train = pos_train + neg_train
    valid = pos_valid + neg_valid
    test  = pos_test  + neg_test
    random.shuffle(train); random.shuffle(valid); random.shuffle(test)

    stats = {
        "selected": {"pos": len(pos_sel), "neg": len(neg_sel)},
        "train": {"pos": len(pos_train), "neg": len(neg_train), "total": len(train)},
        "valid": {"pos": len(pos_valid), "neg": len(neg_valid), "total": len(valid)},
        "test":  {"pos": len(pos_test),  "neg": len(neg_test),  "total": len(test)},
    }
    return train, valid, test, stats

def copy_pair(img: Path, lbl: Path, out_img: Path, out_lbl: Path):
    ensure_dir(out_img.parent)
    ensure_dir(out_lbl.parent)
    shutil.copy2(img, out_img)
    if lbl.exists():
        shutil.copy2(lbl, out_lbl)
    else:
        with open(out_lbl, "w", encoding="utf-8"):
            pass

def build_dataset(
    pos_root: str = "data/autolabeled_yolo",
    neg_root: str = "data/negatives_only",
    out_root: str = "data/mixed_yolo_25_70_20_10",
    seed: int = 42
):
    pos_root = Path(pos_root)
    neg_root = Path(neg_root)
    out_root = Path(out_root)

    print("[INFO] читаю позитивы из", pos_root)
    positives = collect_positives(pos_root)
    print(f"[INFO] позитивов (непустых .txt): {len(positives)}")

    print("[INFO] читаю негативы из", neg_root)
    negatives = collect_negatives(neg_root)
    print(f"[INFO] негативов: {len(negatives)}")

    train, valid, test, stats = stratified_mix(positives, negatives, seed=seed, target_pos_ratio=0.25)
    print("[INFO] план:", stats)

    for sub in ["images/train","images/valid","images/test","labels/train","labels/valid","labels/test"]:
        ensure_dir(out_root / sub)

    def put_split(pairs: List[Tuple[Path, Path]], split: str, prefix: str):
        for img, lbl in pairs:
            rel = Path(prefix) / img.name  # компактно: имя файла + префикс pos/neg
            out_img = out_root / "images" / split / rel
            out_lbl = out_root / "labels" / split / rel.with_suffix(".txt")
            copy_pair(img, lbl, out_img, out_lbl)

    pos_train = [(i,l) for (i,l) in train if non_empty_file(l)]
    neg_train = [(i,l) for (i,l) in train if not non_empty_file(l)]
    pos_valid = [(i,l) for (i,l) in valid if non_empty_file(l)]
    neg_valid = [(i,l) for (i,l) in valid if not non_empty_file(l)]
    pos_test  = [(i,l) for (i,l) in test  if non_empty_file(l)]
    neg_test  = [(i,l) for (i,l) in test  if not non_empty_file(l)]

    put_split(pos_train, "train", "pos")
    put_split(neg_train, "train", "neg")
    put_split(pos_valid, "valid", "pos")
    put_split(neg_valid, "valid", "neg")
    put_split(pos_test,  "test",  "pos")
    put_split(neg_test,  "test",  "neg")

    class_names = read_classes(pos_root)
    (out_root / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")

    data_yaml = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names
    }
    with open(out_root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    print("\n[DONE] новый датасет:", out_root.resolve())
    print("  train images:", len(list((out_root / 'images/train').rglob('*'))))
    print("  valid images:", len(list((out_root / 'images/valid').rglob('*'))))
    print("  test  images:", len(list((out_root / 'images/test').rglob('*'))))
    print("  data.yaml   :", (out_root / 'data.yaml').resolve())

if __name__ == "__main__":
    build_dataset(
        pos_root="data/autolabeled_yolo",
        neg_root="data/negatives_only",
        out_root="data/mixed_yolo_25_70_20_10",
        seed=42
    )