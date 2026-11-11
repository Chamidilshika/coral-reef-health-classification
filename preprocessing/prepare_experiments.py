from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_DIR / "dataset" / "sources"
OUTPUT_DIR = PROJECT_DIR / "dataset" / "experiments"

CLASSES = ["healthy", "bleached"]
SOURCES = ["kaggle", "self_collected"]
SPLITS = ["train", "val", "test"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def copy_files(files, source, category, split):
    destination = OUTPUT_DIR / f"{source}_only" / split / category
    destination.mkdir(parents=True, exist_ok=True)

    for path in files:
        target = destination / f"{source}_{path.name}"
        shutil.copy2(path, target)

def split_source(source):
    for category in CLASSES:
        folder = SOURCE_DIR / source / category

        if not folder.exists():
            raise FileNotFoundError(
                f"Missing folder: {folder}"
            )

        files = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )

        if len(files) < 3:
            raise ValueError(
                f"Not enough images in {folder}"
            )

        train_files, temp_files = train_test_split(
            files,
            test_size=0.30,
            random_state=42
        )

        val_files, test_files = train_test_split(
            temp_files,
            test_size=0.50,
            random_state=42
        )

        for split, split_files in [
            ("train", train_files),
            ("val", val_files),
            ("test", test_files)
        ]:
            copy_files(
                split_files, source, category, split
            )

        print(
            f"{source} / {category}: "
            f"train={len(train_files)}, "
            f"val={len(val_files)}, "
            f"test={len(test_files)}"
        )

def build_combined():
    for split in SPLITS:
        for category in CLASSES:
            destination = (
                OUTPUT_DIR / "combined" / split / category
            )
            destination.mkdir(parents=True, exist_ok=True)

            for source in SOURCES:
                source_folder = (
                    OUTPUT_DIR / f"{source}_only"
                    / split / category
                )

                for path in source_folder.iterdir():
                    if path.is_file():
                        shutil.copy2(
                            path, destination / path.name
                        )

    print("\nCombined dataset created.")

if __name__ == "__main__":
    # Run only after cleaning and duplicate review
    for source in SOURCES:
        split_source(source)

    build_combined()
    print("\nAll experiment splits are ready.")
