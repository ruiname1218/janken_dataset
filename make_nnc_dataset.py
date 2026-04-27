"""
Convert Rock-Paper-Scissors images into an NNC (Neural Network Console) dataset.

Usage:
  python make_nnc_dataset.py            # default: 64x64 -> ./nnc_dataset/
  python make_nnc_dataset.py --size 32  # 32x32 -> ./nnc_dataset_32/

Output layout:
  ./nnc_dataset[_<size>]/
    train/{paper,rock,scissors}/*.png
    test/{paper,rock,scissors}/*.png
    train.csv
    test.csv

CSV format:
  x:image,y:label
  ./train/paper/xxx.png,0
  ...

Labels: paper=0, rock=1, scissors=2
"""

import argparse
import csv
from pathlib import Path
from PIL import Image

SRC_ROOT = Path(__file__).parent
LABELS = {"paper": 0, "rock": 1, "scissors": 2}
SPLITS = ["train", "test"]


def convert_image(src: Path, dst: Path, size: tuple[int, int]) -> None:
    img = Image.open(src)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        rgba = img.convert("RGBA")
        bg.paste(rgba, mask=rgba.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")
    img = img.resize(size, Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, format="PNG")


def process_split(split: str, dst_root: Path, size: tuple[int, int]) -> None:
    src_split = SRC_ROOT / split
    dst_split = dst_root / split
    rows = []

    for class_name, label in LABELS.items():
        src_dir = src_split / class_name
        if not src_dir.is_dir():
            print(f"  skip: {src_dir} not found")
            continue
        files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() == ".png")
        for src_path in files:
            dst_path = dst_split / class_name / src_path.name
            convert_image(src_path, dst_path, size)
            rel = f"./{split}/{class_name}/{src_path.name}"
            rows.append((rel, label))
        print(f"  {split}/{class_name}: {len(files)} images")

    csv_path = dst_root / f"{split}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x:image", "y:label"])
        writer.writerows(rows)
    print(f"  wrote {csv_path} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=64,
                        help="output image size (square). default: 64")
    parser.add_argument("--out", type=str, default=None,
                        help="output dir. default: nnc_dataset (size=64) "
                             "or nnc_dataset_<size> otherwise")
    args = parser.parse_args()

    size = (args.size, args.size)
    if args.out is not None:
        dst_root = SRC_ROOT / args.out
    elif args.size == 64:
        dst_root = SRC_ROOT / "nnc_dataset"
    else:
        dst_root = SRC_ROOT / f"nnc_dataset_{args.size}"

    dst_root.mkdir(exist_ok=True)
    print(f"Generating {size[0]}x{size[1]} dataset at {dst_root}")
    for split in SPLITS:
        print(f"[{split}]")
        process_split(split, dst_root, size)
    print("\nDone.")
    print(f"Output: {dst_root}")


if __name__ == "__main__":
    main()
