# -*- coding: utf-8 -*-
# @Time    : 2025/9/1 17:30:35
# @Author  : 墨烟行(GitHub UserName: CloudSwordSage)
# @File    : loss.py
# @Desc    : 损失函数

import torch.nn as nn
import torch
import numpy as np


class JointsMSELoss(nn.Module):
    def __init__(self):
        super(JointsMSELoss, self).__init__()

    def forward(self, output, target, weight=None):
        pred = output.view(output.size(0), output.size(1), -1)
        gt = target.view(target.size(0), target.size(1), -1)

        loss = (pred - gt) ** 2

        if weight is not None:
            w = weight.view(weight.size(0), weight.size(1), -1)
            loss = loss * w

        loss = loss.mean(dim=2).mean(dim=1)
        return loss.mean()


def generate_heatmap_and_weight(
    keypoints, image_size=(256, 192), heatmap_size=(64, 48), sigma=2.0, min_weight=0.2
):
    num_kps = keypoints.shape[0]
    heatmaps = np.zeros((num_kps, heatmap_size[0], heatmap_size[1]), dtype=np.float32)
    weights = np.zeros_like(heatmaps, dtype=np.float32)

    img_h, img_w = image_size
    kps_abs = keypoints * np.array([img_w, img_h])

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
        weights[i] = min_weight + (1.0 - min_weight) * gaussian

    return heatmaps, weights


if __name__ == "__main__":
    criterion = JointsMSELoss()

    keypoints = np.random.rand(21, 2)

    heatmaps, weights = generate_heatmap_and_weight(keypoints)
    target = torch.tensor(heatmaps, dtype=torch.float32).unsqueeze(0)
    weight = torch.tensor(weights, dtype=torch.float32).unsqueeze(0)

    pred_random = torch.rand_like(target)
    loss_random = criterion(pred_random, target, weight)

    pred_perfect = target.clone()
    loss_perfect = criterion(pred_perfect, target, weight)

    print(f"初始输出 (随机 vs GT) 损失: {loss_random:.6f}")
    print(f"收敛输出 (GT vs GT) 损失: {loss_perfect:.6f}")
