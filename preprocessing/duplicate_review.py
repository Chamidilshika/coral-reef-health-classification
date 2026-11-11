
from pathlib import Path
from collections import defaultdict
from PIL import Image
import hashlib
import csv

PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_DIR / "dataset" / "sources"
REPORT_PATH = PROJECT_DIR / "dataset" / "duplicate_review.csv"

SOURCES = ["kaggle", "self_collected"]
CLASSES = ["healthy", "bleached"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

groups = defaultdict(list)

for source in SOURCES:
    for category in CLASSES:
        folder = SOURCE_DIR / source / category

        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue

            try:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    image.load()
                    digest = hashlib.sha256(
                        str(image.size).encode() + image.tobytes()
                    ).hexdigest()

                groups[digest].append((source, category, str(path)))

            except Exception as error:
                print(f"Could not read {path}: {error}")

rows = []

for digest, files in groups.items():
    if len(files) < 2:
        continue

    labels = {category for _, category, _ in files}
    status = (
        "REVIEW_CONFLICTING_LABELS"
        if len(labels) > 1
        else "REVIEW_SAME_LABEL_DUPLICATES"
    )

    for source, category, path in files:
        rows.append({
            "duplicate_group": digest[:12],
            "source": source,
            "class_label": category,
            "review_status": status,
            "file_path": path,
        })

with REPORT_PATH.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "duplicate_group",
            "source",
            "class_label",
            "review_status",
            "file_path",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Duplicate review report saved to: {REPORT_PATH}")
print(f"Files included in duplicate groups: {len(rows)}")
print("No images were deleted or modified.")
