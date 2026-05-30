from pathlib import Path
from collections import Counter

SOURCE_BASE = r"D:\UNI\Sem6\Machine Learning\Project\dataset\dataset"
SECTIONS = ["sec1", "sec2", "sec3", "sec4"]


def collect_class_counts(base_path, sections):
    class_counts = Counter()
    for section in sections:
        section_path = Path(base_path) / section
        if not section_path.exists():
            print(f"⚠️  Missing folder: {section_path}")
            continue
        for label_path in section_path.glob("*.txt"):
            try:
                with label_path.open("r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            class_counts[class_id] += 1
            except Exception:
                continue
    return class_counts


def print_sample_images(base_path, sections, limit=5):
    for section in sections:
        section_path = Path(base_path) / section
        if not section_path.exists():
            continue
        images = sorted([p.name for p in section_path.glob("*.png")])[:limit]
        print(f"{section} first {limit} images: {images}")


def main():
    class_counts = collect_class_counts(SOURCE_BASE, SECTIONS)
    if class_counts:
        summary = ", ".join([f"{k}: {v} times" for k, v in sorted(class_counts.items())])
        print(f"Class IDs found: {{{summary}}}")
    else:
        print("Class IDs found: {}")

    print_sample_images(SOURCE_BASE, SECTIONS, limit=5)


if __name__ == "__main__":
    main()
