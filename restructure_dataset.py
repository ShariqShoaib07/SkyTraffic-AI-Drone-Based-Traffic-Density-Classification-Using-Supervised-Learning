import random
import shutil
from pathlib import Path

SOURCE_BASE = r"D:\UNI\Sem6\Machine Learning\Project\dataset\dataset"
DEST_BASE = r"D:\UNI\Sem6\Machine Learning\Project\YOLODataset"
SECTIONS = ["sec1", "sec2", "sec3", "sec4", "sec5", "sec6", "sec7",
            "sec8", "sec9", "sec_a", "sec_b", "sec_c"]
TRAIN_SPLIT = 0.85

def collect_pairs(source_base, sections):
    pairs = []
    missing_labels = []
    sections_found = []
    sections_missing = []
    base_path = Path(source_base)
    for section in sections:
        section_path = base_path / section
        if not section_path.exists():
            print(f"Skipping {section} -- folder not found")
            sections_missing.append(section)
            continue
        sections_found.append(section)
        for img_path in section_path.glob("*.png"):
            label_path = img_path.with_suffix(".txt")
            if label_path.exists():
                pairs.append((img_path, label_path))
            else:
                missing_labels.append(img_path)
    return pairs, missing_labels, sections_found, sections_missing

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
    pairs, missing_labels, sections_found, sections_missing = collect_pairs(SOURCE_BASE, SECTIONS)
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
    print("\n" + "="*50)
    print("DATASET RESTRUCTURE SUMMARY")
    print("="*50)
    print(f"Sections found:     {len(sections_found)} / 12")
    print(f"Sections found:     {', '.join(sections_found) if sections_found else 'None'}")
    if sections_missing:
        print(f"Sections missing:   {', '.join(sections_missing)}")
    print(f"Total image pairs:  {len(pairs)}")
    print(f"Train pairs:        {len(train_pairs)}")
    print(f"Val pairs:          {len(val_pairs)}")
    print(f"Skipped (no .txt):  {len(missing_labels)} images")
    print("="*50)

if __name__ == "__main__":
    main()
