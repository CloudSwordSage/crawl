# -*- coding: utf-8 -*-
# @Time    : 2025/9/1 14:19:26
# @Author  : 墨烟行(GitHub UserName: CloudSwordSage)
# @File    : data.py
# @Desc    : 加载 COCO KeyPoints 数据集

import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from typing import List, Dict, Any, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2

TRAIN_JSON = None
TRAIN_IMG_PATH = "./datasets/HandKeypoints/trains/"
VAL_IMG_PATH = "./datasets/HandKeypoints/vals/"
VAL_JSON = None

with open("./datasets/HandKeypoints/train.json", "r") as f:
    TRAIN_JSON = json.load(f)

with open("./datasets/HandKeypoints/val.json", "r") as f:
    VAL_JSON = json.load(f)

background = [cv2.imread(f"./background/img_{i}.jpg") for i in range(1, 14)]
background = [bg for bg in background if bg is not None]


class HandKeypointsDataset(Dataset):
    def __init__(
        self,
        json_data: Dict[str, Any],
        background_images: List[np.ndarray],
        image_size: tuple = (256, 192),
        is_train: bool = True,
        use_augmentation: bool = True,
    ):
        self.json_data = json_data
        self.background_images = background_images
        self.image_size = image_size  # (height, width)
        self.is_train = is_train
        self.use_augmentation = use_augmentation
        self.images = json_data.get("images", [])
        self.annotations = json_data.get("annotations", [])

        self.image_id_to_annotations = {}
        for ann in self.annotations:
            image_id = ann["image_id"]
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(ann)

        self.image_id_to_info = {img["id"]: img for img in self.images}

        self.transform = self._get_transforms()

    def _get_transforms(self):
        if self.is_train and self.use_augmentation:
            return A.Compose(
                [
                    # 空间变换
                    A.HorizontalFlip(p=0.5),
                    A.Affine(rotate=(-30, 30), p=0.5, keep_ratio=True),
                    A.Affine(scale=(0.8, 1.2), p=0.5, keep_ratio=True),
                    A.Affine(
                        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                        scale=(0.8, 1.2),
                        rotate=(-30, 30),
                        p=0.7,
                        keep_ratio=True,
                    ),
                    # 颜色变换
                    A.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                    ),
                    A.GaussianBlur(blur_limit=3, p=0.3),
                    # 归一化
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ],
                keypoint_params=A.KeypointParams(
                    format="xy", remove_invisible=False, angle_in_degrees=True
                ),
            )
        else:
            return A.Compose(
                [
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ],
                keypoint_params=A.KeypointParams(
                    format="xy", remove_invisible=False, angle_in_degrees=True
                ),
            )

    def _replace_green_screen(self, image: np.ndarray) -> np.ndarray:
        if len(self.background_images) == 0:
            return image

        bg = self.background_images[np.random.randint(0, len(self.background_images))]

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])

        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.bitwise_not(mask)

        bg = cv2.resize(bg, (image.shape[1], image.shape[0]))

        result = cv2.bitwise_and(image, image, mask=mask)
        background_part = cv2.bitwise_and(bg, bg, mask=cv2.bitwise_not(mask))
        result = cv2.add(result, background_part)

        return result

    def __resize_img(self, image_tensor: torch.Tensor, keypoints: np.ndarray):
        _, orig_h, orig_w = image_tensor.shape
        target_h, target_w = self.image_size

        scale_h = target_h / orig_h
        scale_w = target_w / orig_w
        scale = min(scale_h, scale_w)

        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)

        resized_image = torch.nn.functional.interpolate(
            image_tensor.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        padded_image = torch.zeros((3, target_h, target_w), dtype=image_tensor.dtype)

        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2

        padded_image[:, pad_top : pad_top + new_h, pad_left : pad_left + new_w] = (
            resized_image
        )

        resized_keypoints = []
        for x, y in keypoints:
            x_scaled = x * scale
            y_scaled = y * scale

            x_final = x_scaled + pad_left
            y_final = y_scaled + pad_top

            x_normalized = x_final / target_w
            y_normalized = y_final / target_h

            resized_keypoints.append([x_normalized, y_normalized])

        return padded_image, np.array(resized_keypoints, dtype=np.float32)

    def _generate_heatmap(self, keypoints_norm, heatmap_size=(64, 48), sigma=2.0):
        num_kps = keypoints_norm.shape[0]
        heatmaps = np.zeros(
            (num_kps, heatmap_size[0], heatmap_size[1]), dtype=np.float32
        )
        weights = np.zeros_like(heatmaps, dtype=np.float32)

        img_h, img_w = self.image_size
        kps_abs = keypoints_norm * np.array([img_w, img_h])

        heatmap_h, heatmap_w = heatmap_size
        scale_h = heatmap_h / img_h
        scale_w = heatmap_w / img_w
        kps_heatmap = kps_abs * np.array([scale_w, scale_h])

        for i in range(num_kps):
            x, y = kps_heatmap[i]
            if x < 0 or x >= heatmap_w or y < 0 or y >= heatmap_h:
                continue

            xx, yy = np.meshgrid(np.arange(heatmap_w), np.arange(heatmap_h))
            gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
            gaussian = gaussian / np.max(gaussian)
            heatmaps[i] = gaussian

            weights[i] = 0.2 + (1.0 - 0.2) * gaussian

        return heatmaps, weights

    def _calculate_area_from_keypoints(self, keypoints: np.ndarray) -> float:
        if len(keypoints) == 0:
            return 0.0

        valid_kps = keypoints[np.any(keypoints > 0.01, axis=1)]
        if len(valid_kps) < 2:
            return 0.0

        min_x, min_y = np.min(valid_kps, axis=0)
        max_x, max_y = np.max(valid_kps, axis=0)

        width = max(1.0, max_x - min_x)
        height = max(1.0, max_y - min_y)

        return width * height

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_id = image_info["id"]

        image_path = image_info["file_name"]
        if self.is_train:
            image = cv2.imread(os.path.join(TRAIN_IMG_PATH, image_path))
        else:
            image = cv2.imread(os.path.join(VAL_IMG_PATH, image_path))
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self._replace_green_screen(image)

        annotations = self.image_id_to_annotations.get(image_id, [])
        if len(annotations) == 0:
            keypoints = np.zeros((21, 2), dtype=np.float32)
            area = 0.0
        else:
            ann_list = []
            for ann in annotations:
                ann_area = ann.get("area", 0.0)
                kps = ann["keypoints"]
                ann_keypoints = []
                for i in range(0, len(kps), 3):
                    x, y, v = kps[i], kps[i + 1], kps[i + 2]
                    if v > 0:
                        ann_keypoints.append((x, y))
                ann_keypoints = np.array(ann_keypoints, dtype=np.float32)

                if ann_area <= 0:
                    ann_area = self._calculate_area_from_keypoints(ann_keypoints)

                ann_list.append((ann, ann_area, ann_keypoints))

            selected_ann, selected_area, selected_kps = max(
                ann_list, key=lambda x: x[1]
            )
            area = selected_area

            keypoints = []
            for x, y in selected_kps:
                keypoints.append((x, y))
            while len(keypoints) < 21:
                keypoints.append((0.0, 0.0))
            keypoints = np.array(keypoints[:21], dtype=np.float32)

        transformed = self.transform(image=image, keypoints=keypoints)
        image_tensor = transformed["image"]
        keypoints_transformed = transformed["keypoints"]

        image_resize, keypoints_resize = self.__resize_img(
            image_tensor, keypoints_transformed
        )

        heatmaps, weights = self._generate_heatmap(
            keypoints_resize, heatmap_size=(64, 48)
        )

        return {
            "image": image_resize,
            "keypoints": torch.tensor(keypoints_resize, dtype=torch.float32),
            "areas": area,
            "heatmaps": torch.tensor(heatmaps, dtype=torch.float32),
            "weights": torch.tensor(weights, dtype=torch.float32),
            "image_id": image_id,
            "original_size": torch.tensor(image.shape[:2]),
        }


def create_data_loaders(
    batch_size: int = 32,
    num_workers: int = 4,
    prefetch_factor=2,
    image_size: tuple = (256, 192),
    use_augmentation: bool = True,
):

    train_dataset = HandKeypointsDataset(
        json_data=TRAIN_JSON,
        background_images=background,
        image_size=image_size,
        is_train=True,
        use_augmentation=use_augmentation,
    )

    val_dataset = HandKeypointsDataset(
        json_data=VAL_JSON,
        background_images=background,
        image_size=image_size,
        is_train=False,
        use_augmentation=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
    )

    return train_loader, val_loader


def __visualize_resized_images(batch, num_samples=8):
    import matplotlib.pyplot as plt
    import seaborn as sns

    images = batch["image"].cpu()
    keypoints = batch["keypoints"].cpu()
    heatmaps = batch["heatmaps"].cpu()
    B, _, H_img, W_img = images.shape

    # fmt: off
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 4],  # 拇指
        [0, 5], [5, 6], [6, 7], [7, 8],  # 食指
        [0, 9], [9, 10], [10, 11], [11, 12],  # 中指
        [0, 13], [13, 14], [14, 15], [15, 16],  # 无名指
        [0, 17], [17, 18], [18, 19], [19, 20],  # 小指
    ]
    # fmt: on

    valid_indices = [i for i in range(B) if keypoints[i].shape == (21, 2)]
    num_display = min(num_samples, len(valid_indices), 8)
    if num_display == 0:
        print("无有效样本（需每个样本含21个关键点）")
        return

    fig = plt.figure(figsize=(12, 12))
    fig.suptitle("Hand Keypoints Visualization", fontsize=16, y=0.98)

    for idx in range(num_display):
        batch_idx = valid_indices[idx]
        row = idx // 2
        col = (idx % 2) * 2

        ax_img = plt.subplot(4, 4, row * 4 + col + 1)

        img = images[batch_idx].permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)

        ax_img.imshow(img)
        ax_img.set_title(f"Sample {batch_idx}", fontsize=10, pad=5)
        ax_img.axis("off")

        kps_norm = keypoints[batch_idx].numpy()
        kps_pixel = kps_norm * np.array([W_img, H_img])
        ax_img.scatter(
            kps_pixel[:, 0],
            kps_pixel[:, 1],
            c="red",
            s=25,
            marker="o",
            zorder=3,
            edgecolors="white",
            linewidths=0.5,
        )

        for p1, p2 in connections:
            x = [kps_pixel[p1, 0], kps_pixel[p2, 0]]
            y = [kps_pixel[p1, 1], kps_pixel[p2, 1]]
            ax_img.plot(x, y, "g-", linewidth=1.5, zorder=2, alpha=0.8)

        ax_hm = plt.subplot(4, 4, row * 4 + col + 2)

        fused_heatmap = heatmaps[batch_idx].max(dim=0).values.numpy()

        fused_heatmap_resized = cv2.resize(
            fused_heatmap, (W_img, H_img), interpolation=cv2.INTER_LINEAR
        )

        im = ax_hm.imshow(fused_heatmap_resized, cmap="jet", vmin=0, vmax=1)
        ax_hm.set_title("Fused Heatmap", fontsize=10, pad=5)
        ax_hm.axis("off")

    cbar_ax = fig.add_axes([0.92, 0.05, 0.015, 0.9])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label("Heatmap Intensity", rotation=270, labelpad=15, fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    plt.subplots_adjust(
        left=0.02, right=0.9, bottom=0.05, top=0.95, wspace=0.05, hspace=0.15
    )
    plt.show()


if __name__ == "__main__":
    from config import HRNetConfig
    from util import get_keypoints, oks, mAP50, mAP50_95

    config = HRNetConfig()
    train_loader, val_loader = create_data_loaders(
        batch_size=16,
        num_workers=4,
        prefetch_factor=1,
        image_size=(config.model.input_height, config.model.input_width),
    )

    img_h, img_w = config.model.input_height, config.model.input_width

    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")

    batch = next(iter(train_loader))
    images = batch["image"]
    keypoints_gt = batch["keypoints"]
    heatmaps = batch["heatmaps"]
    areas = batch["areas"]
    print(f"图像形状: {batch['image'].shape}")
    print(f"关键点形状: {batch['keypoints'].shape}")
    print(f"图像ID: {batch['image_id']}")
    print(f"热力图形状: {batch['heatmaps'].shape}")
    print(
        f"热力图范围： min {heatmaps[0].min().item()}, max {heatmaps[0].max().item()}"
    )

    B, K, H_hm, W_hm = heatmaps.shape
    coords_pred_list = []
    scores_list = []

    for b in range(B):
        coords_pred, scores = get_keypoints(heatmaps[b].numpy(), window_size=5)
        coords_pred_list.append(coords_pred)
        scores_list.append(scores.squeeze(-1))

    coords_pred = np.stack(coords_pred_list)
    scores_pred = np.stack(scores_list)

    keypoints_gt_np = np.stack(
        [k[:, :2].numpy() if hasattr(k, "numpy") else k[:, :2] for k in keypoints_gt]
    )

    print(f"解析后关键点坐标形状: {coords_pred.shape}")
    print(f"解析后置信度形状: {scores_pred.shape}")

    keypoints_gt_pix = keypoints_gt_np * np.array([img_w, img_h])
    coords_pred_pix = coords_pred * np.array([img_w, img_h])

    oks_scores = []
    for b in range(B):
        oks_val = oks(keypoints_gt_pix[b], coords_pred_pix[b], a_g=areas[b].item())
        oks_scores.append(oks_val)
    oks_scores = np.array(oks_scores)
    print(f"OKS 分数: {oks_scores}")

    gts_list = [keypoints_gt_pix[b] for b in range(B)]
    dts_list = [coords_pred_pix[b] for b in range(B)]
    scores_list_per_sample = [scores_pred[b] for b in range(B)]

    map50 = mAP50(gts_list, dts_list, scores_list_per_sample, a_g=areas[b].item())
    map50_95 = mAP50_95(gts_list, dts_list, scores_list_per_sample, a_g=areas[b].item())
    print(f"mAP@50: {map50}")
    print(f"mAP@0.5:0.95: {map50_95}")
