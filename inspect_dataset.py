from pathlib import Path
from collections import Counter

SOURCE_BASE = r"D:\UNI\Sem6\Machine Learning\Project\dataset\dataset"
SECTIONS = ["sec1", "sec2", "sec3", "sec4", "sec5", "sec6", "sec7",
            "sec8", "sec9", "sec_a", "sec_b", "sec_c"]


def collect_class_counts(base_path, sections):
    class_counts = Counter()
    sections_found = []
    sections_missing = []
    for section in sections:
        section_path = Path(base_path) / section
        if not section_path.exists():
            print(f"Skipping {section} -- folder not found")
            sections_missing.append(section)
            continue
        sections_found.append(section)
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
    return class_counts, sections_found, sections_missing


def print_sample_images(base_path, sections, limit=5):
    for section in sections:
        section_path = Path(base_path) / section
        if not section_path.exists():
            continue
        images = sorted([p.name for p in section_path.glob("*.png")])[:limit]
        print(f"{section} first {limit} images: {images}")


def main():
    class_counts, sections_found, sections_missing = collect_class_counts(SOURCE_BASE, SECTIONS)

    print("\n" + "="*60)
    print("DATASET INSPECTION SUMMARY")
    print("="*60)
    print(f"Sections found:     {len(sections_found)} / 12")
    if sections_found:
        print(f"  Present:          {', '.join(sections_found)}")
    if sections_missing:
        print(f"  Missing:          {', '.join(sections_missing)}")

    if class_counts:
        total_detections = sum(class_counts.values())
        summary = ", ".join([f"{k}: {v:,}" for k, v in sorted(class_counts.items())])
        print(f"\nClass IDs found:    {{{summary}}}")
        print(f"Total detections:   {total_detections:,}")
    else:
        print("\nClass IDs found:    {}")

    print("\nSample images:")
    print_sample_images(SOURCE_BASE, sections_found, limit=3)
    print("="*60)


if __name__ == "__main__":
    main()
