import os
import shutil
import random
from pathlib import Path

RAW_DIR = Path("data/raw/tpc-imgs")
PROCESSED_DIR = Path("data/processed")
TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "val"
VAL_SPLIT = 0.2
RANDOM_SEED = 42

def make_dirs(path):
    os.makedirs(path, exist_ok=True)

def main():
    random.seed(RANDOM_SEED)
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    make_dirs(TRAIN_DIR)
    make_dirs(VAL_DIR)

    classes = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    for cls in classes:
        cls_name = cls.name
        images = list(cls.glob("**/*.jpg"))
        random.shuffle(images)

        n_val = int(len(images) * VAL_SPLIT)
        val_imgs = images[:n_val]
        train_imgs = images[n_val:]

        make_dirs(TRAIN_DIR / cls_name)
        make_dirs(VAL_DIR / cls_name)

        for img_path in train_imgs:
            shutil.copy(img_path, TRAIN_DIR / cls_name / img_path.name)
        for img_path in val_imgs:
            shutil.copy(img_path, VAL_DIR / cls_name / img_path.name)

    print(f"Processed dataset ready at {PROCESSED_DIR}")

if __name__ == "__main__":
    main()
