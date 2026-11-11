import os
import shutil
from sklearn.model_selection import train_test_split

RAW_DIR = "dataset/raw"
PROCESSED_DIR = "dataset/processed"

categories = ["healthy", "bleached"]

for category in categories:

    print(os.path.abspath(os.path.join(RAW_DIR, category)))
    images = os.listdir(os.path.join(RAW_DIR,category))

    train, temp = train_test_split(images, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    def move_files(files, split):
        for f in files:
            src = os.path.join(RAW_DIR, category, f)
            dst = os.path.join(PROCESSED_DIR, split, category)

            os.makedirs(dst, exist_ok=True)
            shutil.copy(src, dst)

    move_files(train, "train")
    move_files(val, "val")
    move_files(test, "test")

print("Dataset split completed")