from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_DIR / "dataset" / "sources"

SOURCES = ["kaggle", "self_collected"]

CLASSES = ["healthy", "bleached"]

EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

for source in SOURCES:

    print(f"\nDataset: {source}")

    total = 0

    for category in CLASSES:

        folder = SOURCE_DIR / source / category

        images = [
            file for file in folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in EXTENSIONS
        ]

        count = len(images)

        total += count

        print(f"{category}: {count}")

    print(f"Total: {total}")