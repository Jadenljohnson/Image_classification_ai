# make_split.py
import argparse, random, shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def gather_labeled_files(train_dir: Path):
    """Find files named like cat.0.jpg / dog.1.jpg under train_dir."""
    cats, dogs = [], []
    for p in train_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            name = p.name.lower()
            if name.startswith("cat"):
                cats.append(p)
            elif name.startswith("dog"):
                dogs.append(p)
    return cats, dogs

def copy_split(files, out_train_dir: Path, out_val_dir: Path, val_ratio: float, seed: int):
    random.seed(seed)
    random.shuffle(files)
    k = int(len(files) * val_ratio)
    val = files[:k]
    train = files[k:]
    out_train_dir.mkdir(parents=True, exist_ok=True)
    out_val_dir.mkdir(parents=True, exist_ok=True)
    for s in train:
        shutil.copy2(s, out_train_dir / s.name)
    for s in val:
        shutil.copy2(s, out_val_dir / s.name)
    return len(train), len(val)

def copy_unlabeled_test(test_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for p in test_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            shutil.copy2(p, out_dir / p.name)
            count += 1
    return count

def main():
    ap = argparse.ArgumentParser(description="Create train/val split for Kaggle Dogs vs Cats and stage unlabeled test.")
    ap.add_argument("--kaggle_train", type=str, required=True, help="Path to Kaggle train folder (with cat.* and dog.* files)")
    ap.add_argument("--kaggle_test", type=str, required=True, help="Path to Kaggle test folder (unlabeled, numbered files)")
    ap.add_argument("--out_root", type=str, default="animals", help="Output root to write train/ val/ test_unlabeled/")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_dir = Path(args.kaggle_train)
    test_dir  = Path(args.kaggle_test)
    out_root  = Path(args.out_root)

    cats, dogs = gather_labeled_files(train_dir)
    if not cats or not dogs:
        raise SystemExit(f"Could not find cat.* or dog.* files in: {train_dir}")

    print(f"Found {len(cats)} cat images and {len(dogs)} dog images in train.")

    # Output folders
    out_train_cat = out_root / "train" / "cat"
    out_val_cat   = out_root / "val" / "cat"
    out_train_dog = out_root / "train" / "dog"
    out_val_dog   = out_root / "val" / "dog"
    out_test_unl  = out_root / "test_unlabeled"

    # Split & copy
    cat_tr, cat_val = copy_split(cats, out_train_cat, out_val_cat, args.val_ratio, args.seed)
    dog_tr, dog_val = copy_split(dogs, out_train_dog, out_val_dog, args.val_ratio, args.seed)

    # Copy unlabeled test
    n_test = copy_unlabeled_test(test_dir, out_test_unl)

    print("Split complete.")
    print(f"cat: {cat_tr} train, {cat_val} val")
    print(f"dog: {dog_tr} train, {dog_val} val")
    print(f"Unlabeled test images copied: {n_test}")
    print(f"Output at: {out_root.resolve()}")

if __name__ == "__main__":
    main()
