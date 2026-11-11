
from pathlib import Path
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_DIR / "dataset" / "sources"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for source in ["kaggle", "self_collected"]:
    for category in ["healthy", "bleached"]:
        folder = SOURCE_DIR / source / category

        if not folder.exists():
            print(f"Missing folder: {folder}")
            continue

        for path in folder.iterdir():
            if not path.is_file():
                continue

            if path.suffix.lower() not in EXTENSIONS:
                continue

            try:
                with Image.open(path) as img:
                    img.verify()

            except Exception as error:
                print(f"Corrupted image: {path} | {error}")

print("Image integrity check completed.")
print("Review flagged images manually before removing them.")
