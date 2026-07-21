# data/dataset.py
"""
修复后的木材缺陷分割数据集加载器
主要修复：
1. 修正 ignore_index (255) 被错误 clamp 的问题
2. 规范化标签范围 (0-5 为有效类，255 为忽略类)
3. 严格遵循论文的随机裁剪和缩放策略
"""

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random


class WoodDefectDataset(Dataset):
    """木材缺陷分割数据集 - VOC格式"""

    def __init__(
            self,
            root_dir,
            split='train',
            image_size=512,
            crop_range=(256, 1024),
            augmentation=True,
            ignore_index=255,
            num_classes=6  # Rubber Wood 默认为 6 类 (0-5)
    ):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.crop_range = crop_range
        self.augmentation = augmentation and (split == 'train')
        self.ignore_index = ignore_index
        self.num_classes = num_classes

        self.image_dir = os.path.join(root_dir, 'JPEGImages')
        self.label_dir = os.path.join(root_dir, 'SegmentationClass')

        split_file = os.path.join(root_dir, 'ImageSets', 'Segmentation', f'{split}.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines() if line.strip()]

        # 基础标准化 (DINOv2)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.image_ids)

    def transform(self, image, mask):
        # 1. 随机调整大小 (Random Resize / Crop) - 论文关键步骤
        if self.augmentation:
            # 随机选择一个裁剪尺寸
            target_size = random.randint(self.crop_range[0], self.crop_range[1])

            # 随机裁剪或填充
            i, j, h, w = T.RandomResizedCrop.get_params(
                image, scale=(0.5, 2.0), ratio=(0.75, 1.33)
            )
            image = TF.resized_crop(image, i, j, h, w, (self.image_size, self.image_size))
            mask = TF.resized_crop(mask, i, j, h, w, (self.image_size, self.image_size),
                                   interpolation=TF.InterpolationMode.NEAREST)

            # 2. 随机水平翻转
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # 3. 颜色抖动 (仅图像)
            if random.random() > 0.8:
                color_jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
                image = color_jitter(image)
        else:
            # 测试模式：直接缩放到固定大小
            image = TF.resize(image, (self.image_size, self.image_size))
            mask = TF.resize(mask, (self.image_size, self.image_size),
                             interpolation=TF.InterpolationMode.NEAREST)

        # 转换为 Tensor
        image = TF.to_tensor(image)
        image = self.normalize(image)

        mask = np.array(mask)
        mask = torch.from_numpy(mask).long()

        # ================= 核心修复代码 =================
        # 确保 255 (ignore_index) 被保留，不要被 clamp 掉
        # 只对非 255 的像素进行合法性校验
        valid_mask = (mask != self.ignore_index)

        # 如果存在超出 [0, num_classes-1] 范围的像素（且不是255）
        invalid_mask = valid_mask & (mask >= self.num_classes)
        if invalid_mask.any():
            # 将异常像素统一归类为背景 (0)
            mask[invalid_mask] = 0

        # ===============================================

        return image, mask

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f'{img_id}.jpg')
        mask_path = os.path.join(self.label_dir, f'{img_id}.png')

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        image, mask = self.transform(image, mask)

        return {
            'image': image,
            'label': mask,
            'id': img_id
        }


def create_dataloader(
        root_dir,
        split='train',
        batch_size=4,
        num_workers=4,
        image_size=512,
        crop_range=(256, 1024),
        augmentation=True,
        num_classes=6
):
    dataset = WoodDefectDataset(
        root_dir=root_dir,
        split=split,
        image_size=image_size,
        crop_range=crop_range,
        augmentation=augmentation,
        num_classes=num_classes
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )

    return dataloader