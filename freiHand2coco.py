# -*- coding: utf-8 -*-
# @Time    : 2025/8/31 15:57:03
# @Author  : 墨烟行(GitHub UserName: CloudSwordSage)
# @File    : freiHand2coco.py
# @Desc    : freiHand 数据集转 COCO 格式


import os
import numpy as np
import shutil
import cv2

import json

from copy import deepcopy

ROOT = "./datasets/FreiHAND/original/"
TARGET = "./datasets/HandKeypoints/"

shutil.rmtree(TARGET, ignore_errors=True)

os.makedirs(TARGET, exist_ok=True)
os.makedirs(os.path.join(TARGET, "trains"), exist_ok=True)
os.makedirs(os.path.join(TARGET, "vals"), exist_ok=True)


def projection2uv(xyz, K):
    if not isinstance(xyz, np.ndarray):
        xyz = np.array(xyz)
    if not isinstance(K, np.ndarray):
        K = np.array(K)

    uvw = np.matmul(K, xyz.T).T

    uv = uvw[:, :2] / uvw[:, -1:]

    return uv


train_coco_dct = {
    "images": [],
    "annotations": [],
    "categories": [
        {
            "id": 1,
            "name": "hand",
            # fmt: off
            "keypoints": [
                "wrist",  # 手掌根
                "thumb1", "thumb2", "thumb3", "thumb4",  # 大拇指
                "index1", "index2", "index3", "index4",  # 食指
                "middle1", "middle2", "middle3", "middle4",  # 中指
                "ring1", "ring2", "ring3", "ring4",  # 无名指
                "pinky1", "pinky2", "pinky3","pinky4",  # 小拇指
            ],
            "skeleton": [
                [0, 1], [1, 2], [2, 3], [3, 4],
                [0, 5], [5, 6], [6, 7], [7, 8],
                [0, 9], [9, 10], [10, 11], [11, 12],
                [0, 13], [13, 14], [14, 15], [15, 16],
                [0, 17], [17, 18], [18, 19], [19, 20],
            ],
            # fmt: on
        }
    ],
}

val_coco_dct = deepcopy(train_coco_dct)

train_split = 0.8
val_split = 1 - train_split

with open(os.path.join(ROOT, "training_K.json"), "r") as f:
    K = json.load(f)

with open(os.path.join(ROOT, "training_xyz.json"), "r") as f:
    xyz = json.load(f)

db_data_anno = list(zip(K, xyz))

total = 32560

train_total = int(total * train_split)
val_total = total - train_total

n = 0
while n < total:
    file_name = f"{n:08d}.jpg"
    file_path = os.path.join(ROOT, f"training/rgb/{file_name}")
    img = cv2.imread(file_path)
    h, w = img.shape[:2]

    k, xyz = db_data_anno[n]
    uv = projection2uv(xyz, k)

    x_min, y_min = float(uv[:, 0].min()), float(uv[:, 1].min())
    x_max, y_max = float(uv[:, 0].max()), float(uv[:, 1].max())

    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
    area = (x_max - x_min) * (y_max - y_min)

    keypoints = [coord for (x, y) in uv for coord in (float(x), float(y), 2)]

    if n < train_total:
        train_coco_dct["images"].append(
            {"file_name": file_name, "id": n, "width": w, "height": h}
        )
        ann = {
            "image_id": n,
            "id": n,
            "category_id": 1,
            "keypoints": keypoints,
            "num_keypoints": 21,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
        }
        train_coco_dct["annotations"].append(ann)
        shutil.copy(file_path, os.path.join(TARGET, "trains", file_name))
    else:
        val_coco_dct["images"].append(
            {"file_name": file_name, "id": n, "width": w, "height": h}
        )
        ann = {
            "image_id": n,
            "id": n,
            "category_id": 1,
            "keypoints": keypoints,
            "num_keypoints": 21,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
        }
        val_coco_dct["annotations"].append(ann)
        shutil.copy(file_path, os.path.join(TARGET, "vals", file_name))
    n += 1


with open(os.path.join(TARGET, "train.json"), "w") as f:
    json.dump(train_coco_dct, f, indent=4)

with open(os.path.join(TARGET, "val.json"), "w") as f:
    json.dump(val_coco_dct, f, indent=4)

print("Frei Hand 装换为 COCO 格式完成!")
