import os
import torch
import numpy as np
from PIL import Image
from glob import glob


def check_rubber_dataset():
    root_dir = r'/home/user4/桌面/wood-defect/pine and rubber dataset/rubber dataset'
    label_dir = os.path.join(root_dir, 'SegmentationClass')

    print(f"Checking labels in: {label_dir}")

    label_files = glob(os.path.join(label_dir, '*.png'))
    print(f"Found {len(label_files)} label files")

    all_unique_values = set()
    problem_files = []

    for i, label_path in enumerate(label_files):
        label = np.array(Image.open(label_path))
        unique_values = np.unique(label)
        all_unique_values.update(unique_values.tolist())

        if unique_values.max() >= 6 or unique_values.min() < 0:
            problem_files.append({
                'file': os.path.basename(label_path),
                'min': unique_values.min(),
                'max': unique_values.max(),
                'unique': unique_values.tolist()
            })

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(label_files)}...")

    print(f"\n📊 All unique label values in dataset: {sorted(all_unique_values)}")
    print(f"Expected for Rubber: [0, 1, 2, 3, 4, 5] (6 classes)")

    if problem_files:
        print(f"\n⚠️ Found {len(problem_files)} files with invalid labels:")
        for pf in problem_files[:20]:
            print(f"  {pf['file']}: range [{pf['min']}, {pf['max']}]")
            print(f"    Unique values: {pf['unique']}")
    else:
        print("\n✅ All labels are within valid range!")


if __name__ == "__main__":
    check_rubber_dataset()