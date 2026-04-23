"""
MODIFIED: paper-aligned wood defect dataset pipeline.

Training:
- random short-side resize in [256, 1024]
- random crop/pad to 512 x 512
- horizontal flip
- light color jitter on the image only

Validation / inference:
- direct resize to 512 x 512
"""

import os
import random

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


class WoodDefectDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        image_size=512,
        crop_range=(256, 1024),
        augmentation=True,
        ignore_index=255,
        num_classes=6,
    ):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.crop_range = crop_range
        self.augmentation = augmentation and split == "train"
        self.ignore_index = ignore_index
        self.num_classes = num_classes

        self.image_dir = os.path.join(root_dir, "JPEGImages")
        self.label_dir = os.path.join(root_dir, "SegmentationClass")

        split_file = os.path.join(root_dir, "ImageSets", "Segmentation", f"{split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, "r", encoding="utf-8") as f:
            self.image_ids = [line.strip() for line in f.readlines() if line.strip()]

        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.image_ids)

    def _resize_by_short_side(self, image, mask, short_side):
        width, height = image.size
        scale = short_side / min(width, height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))

        image = TF.resize(image, (new_height, new_width))
        mask = TF.resize(mask, (new_height, new_width), interpolation=TF.InterpolationMode.NEAREST)
        return image, mask

    def _crop_or_pad(self, image, mask):
        width, height = image.size
        target = self.image_size

        if height < target or width < target:
            pad_h = max(0, target - height)
            pad_w = max(0, target - width)
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            image = TF.pad(image, [left, top, right, bottom], fill=0)
            mask = TF.pad(mask, [left, top, right, bottom], fill=self.ignore_index)
            width, height = image.size

        if self.augmentation:
            top = random.randint(0, height - target)
            left = random.randint(0, width - target)
        else:
            top = max(0, (height - target) // 2)
            left = max(0, (width - target) // 2)

        image = TF.crop(image, top, left, target, target)
        mask = TF.crop(mask, top, left, target, target)
        return image, mask

    def transform(self, image, mask):
        if self.augmentation:
            # MODIFIED: actually use the paper's short-side resize range.
            target_short_side = random.randint(self.crop_range[0], self.crop_range[1])
            image, mask = self._resize_by_short_side(image, mask, target_short_side)
            image, mask = self._crop_or_pad(image, mask)

            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() > 0.8:
                color_jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
                image = color_jitter(image)
        else:
            image = TF.resize(image, (self.image_size, self.image_size))
            mask = TF.resize(mask, (self.image_size, self.image_size), interpolation=TF.InterpolationMode.NEAREST)

        image = TF.to_tensor(image)
        image = self.normalize(image)

        mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        valid_mask = mask != self.ignore_index
        invalid_mask = valid_mask & ((mask < 0) | (mask >= self.num_classes))
        if invalid_mask.any():
            mask = mask.clone()
            mask[invalid_mask] = 0

        return image, mask

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        mask_path = os.path.join(self.label_dir, f"{img_id}.png")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        image, mask = self.transform(image, mask)
        return {"image": image, "label": mask, "id": img_id}


def create_dataloader(
    root_dir,
    split="train",
    batch_size=4,
    num_workers=4,
    image_size=512,
    crop_range=(256, 1024),
    augmentation=True,
    num_classes=6,
    ignore_index=255,
):
    dataset = WoodDefectDataset(
        root_dir=root_dir,
        split=split,
        image_size=image_size,
        crop_range=crop_range,
        augmentation=augmentation,
        ignore_index=ignore_index,
        num_classes=num_classes,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
