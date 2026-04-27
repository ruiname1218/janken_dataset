"""
Convert Rock-Paper-Scissors images into an NNC (Neural Network Console) dataset.

Output:
  ./nnc_dataset/
    train/{paper,rock,scissors}/*.png   (64x64 RGB)
    test/{paper,rock,scissors}/*.png    (64x64 RGB)
    train.csv
    test.csv

CSV format:
  x:image,y:label
  ./train/paper/xxx.png,0
  ...

Labels: paper=0, rock=1, scissors=2
"""

import csv
from pathlib import Path
from PIL import Image

SRC_ROOT = Path(__file__).parent
DST_ROOT = SRC_ROOT / "nnc_dataset"
IMG_SIZE = (64, 64)
LABELS = {"paper": 0, "rock": 1, "scissors": 2}
SPLITS = ["train", "test"]


def convert_image(src: Path, dst: Path) -> None:
    img = Image.open(src)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        rgba = img.convert("RGBA")
        bg.paste(rgba, mask=rgba.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, format="PNG")


def process_split(split: str) -> None:
    src_split = SRC_ROOT / split
    dst_split = DST_ROOT / split
    rows = []

    for class_name, label in LABELS.items():
        src_dir = src_split / class_name
        if not src_dir.is_dir():
            print(f"  skip: {src_dir} not found")
            continue
        files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() == ".png")
        for src_path in files:
            dst_path = dst_split / class_name / src_path.name
            convert_image(src_path, dst_path)
            rel = f"./{split}/{class_name}/{src_path.name}"
            rows.append((rel, label))
        print(f"  {split}/{class_name}: {len(files)} images")

    csv_path = DST_ROOT / f"{split}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x:image", "y:label"])
        writer.writerows(rows)
    print(f"  wrote {csv_path} ({len(rows)} rows)")


def main() -> None:
    DST_ROOT.mkdir(exist_ok=True)
    for split in SPLITS:
        print(f"[{split}]")
        process_split(split)
    print("\nDone.")
    print(f"Output: {DST_ROOT}")


if __name__ == "__main__":
    main()
