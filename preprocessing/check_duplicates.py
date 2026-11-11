from pathlib import Path
from collections import defaultdict
from PIL import Image
import hashlib

PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_DIR / "dataset" / "sources"

SOURCES = ["kaggle", "self_collected"]
CLASSES = ["healthy", "bleached"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_image_hash(path):
    """Hash decoded RGB pixels to detect exact visual duplicates."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.load()

        digest = hashlib.sha256()
        digest.update(str(image.size).encode("utf-8"))
        digest.update(image.tobytes())

        return digest.hexdigest()


def main():
    image_groups = defaultdict(list)
    unreadable_images = []

    for source in SOURCES:
        for category in CLASSES:
            folder = SOURCE_DIR / source / category

            if not folder.exists():
                print(f"Missing folder: {folder}")
                continue

            for path in sorted(folder.iterdir()):
                if not path.is_file():
                    continue

                if path.suffix.lower() not in EXTENSIONS:
                    continue

                try:
                    image_hash = get_image_hash(path)
                    image_groups[image_hash].append(
                        (source, category, path)
                    )
                except Exception as error:
                    unreadable_images.append(path)
                    print(f"Cannot read: {path} | {error}")

    duplicate_groups = [
        files for files in image_groups.values()
        if len(files) > 1
    ]

    print("\n========== DUPLICATE CHECK ==========")
    print(f"Unique image groups: {len(image_groups)}")
    print(f"Duplicate groups: {len(duplicate_groups)}")

    for number, group in enumerate(duplicate_groups, start=1):
        print(f"\nDuplicate group {number}:")

        for source, category, path in group:
            print(f"  Source: {source} | Class: {category}")
            print(f"  File: {path}")

        labels = {category for _, category, _ in group}

        if len(labels) > 1:
            print("  WARNING: Matching images have different labels!")

    print(f"\nUnreadable images: {len(unreadable_images)}")
    print("Check completed. No files were deleted or changed.")


if __name__ == "__main__":
    main()
