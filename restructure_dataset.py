import random
import shutil
from pathlib import Path

SOURCE_BASE = r"D:\UNI\Sem6\Machine Learning\Project\dataset\dataset"
DEST_BASE = r"D:\UNI\Sem6\Machine Learning\Project\YOLODataset"
SECTIONS = ["sec1", "sec2", "sec3", "sec4", "sec5", "sec6", "sec7"]
TRAIN_SPLIT = 0.85

def collect_pairs(source_base, sections):
    pairs = []
    missing_labels = []
    base_path = Path(source_base)
    for section in sections:
        section_path = base_path / section
        if not section_path.exists():
            print(f"??  Missing folder: {section_path}")
            continue
        for img_path in section_path.glob("*.png"):
            label_path = img_path.with_suffix(".txt")
            if label_path.exists():
                pairs.append((img_path, label_path))
            else:
                missing_labels.append(img_path)
    return pairs, missing_labels

def ensure_dest_dirs(dest_base):
    dest_path = Path(dest_base)
    train_images = dest_path / "train" / "images"
    train_labels = dest_path / "train" / "labels"
    val_images = dest_path / "val" / "images"
    val_labels = dest_path / "val" / "labels"
    for path in [train_images, train_labels, val_images, val_labels]:
        path.mkdir(parents=True, exist_ok=True)
    return train_images, train_labels, val_images, val_labels

def copy_pairs(pairs, images_dir, labels_dir):
    for idx, (img_path, label_path) in enumerate(pairs, 1):
        shutil.copy2(img_path, images_dir / img_path.name)
        shutil.copy2(label_path, labels_dir / label_path.name)
        if idx % 5000 == 0:
            print(f"   Copied {idx}/{len(pairs)} pairs...")

def main():
    random.seed(42)
    print("Collecting pairs...")
    pairs, missing_labels = collect_pairs(SOURCE_BASE, SECTIONS)
    print(f"Found {len(pairs)} pairs")
    random.shuffle(pairs)
    train_count = int(len(pairs) * TRAIN_SPLIT)
    train_pairs = pairs[:train_count]
    val_pairs = pairs[train_count:]
    train_images, train_labels, val_images, val_labels = ensure_dest_dirs(DEST_BASE)
    print(f"Copying {len(train_pairs)} train pairs...")
    copy_pairs(train_pairs, train_images, train_labels)
    print(f"Copying {len(val_pairs)} val pairs...")
    copy_pairs(val_pairs, val_images, val_labels)
    print("? Dataset restructure complete")
    print(f"Total pairs found: {len(pairs)}")
    print(f"Train count: {len(train_pairs)}")
    print(f"Val count: {len(val_pairs)}")
    if missing_labels:
        print(f"Images without matching .txt labels: {len(missing_labels)}")
    else:
        print("Images without matching .txt labels: None")

if __name__ == "__main__":
    main()
