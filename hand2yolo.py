# -*- coding: utf-8 -*-
# @Time    : 2025/8/30 22:36:21
# @Author  : 墨烟行(GitHub UserName: CloudSwordSage)
# @File    : hand2yolo.py
# @Desc    : 将 COCO-Hand 和 Oxford Hand 数据集转换成 YOLO 格式

"""
COCO-Hand 目录结构：
COCO_HAND_ROOT
├── COCO-Hand-S
│   ├── COCO-Hand-S_Images
│   └── COCO-Hand-S_annotations.txt
└── COCO-Hand-Big
    ├── COCO-Hand-Big_Images
    └── COCO-Hand-Big_annotations.txt
annotations文件格式：每行均为：filename, xmin, xmax, ymin, ymax, x1, y1, x2, y2, x3, y3, x4, y4, class_name
Oxford Hand 目录结构：
OXFORD_HAND_ROOT
├── training_dataset
│   └── training_data
│       ├── annotations
│       │   ├── hand_1.mat
│       │   └── ...
│       └── images
│            ├── hand_1.jpg
│            └── ...
├── test_dataset
│   └── test_data
│       ├── annotations
│       │   ├── hand_1.mat
│       │   └── ...
│       └── images
│            ├── hand_1.jpg
│            └── ...
└── validation_dataset
    └── validation_data
        ├── annotations
        │   ├── hand_1.mat
        │   └── ...
        └── images
             ├── hand_1.jpg
             └── ...
"""


import os
import random
import shutil
import scipy.io as sio
from pathlib import Path
from PIL import Image

COCO_HAND_ROOT = "./datasets/COCOHand/original/"
OXFORD_HAND_ROOT = "./datasets/Oxford/original/"
TARGET = "./datasets/Hands/"

SPLITS_INFO = {"val": 0.7}
SPLITS_INFO["test"] = 1 - SPLITS_INFO["val"]
CLASS_ID = 0
MAX_RETRIES = 3

# ========== Step 0: 全局文件计数器 ==========
file_counter = 0


def get_unique_filename():
    global file_counter
    file_counter += 1
    return f"{file_counter:06d}.jpg"


# ========== Step 1: 清空目标文件夹 ==========
if os.path.exists(TARGET):
    shutil.rmtree(TARGET)
for split in ["train", "val", "test"]:
    (Path(TARGET) / "images" / split).mkdir(parents=True, exist_ok=True)
    (Path(TARGET) / "labels" / split).mkdir(parents=True, exist_ok=True)
print("目标目录初始化完成！")

skipped_files = []

# ========== Step 2: COCO Hand-S ==========
ann_file = os.path.join(COCO_HAND_ROOT, "COCO-Hand-S", "COCO-Hand-S_annotations.txt")
img_dir = os.path.join(COCO_HAND_ROOT, "COCO-Hand-S", "COCO-Hand-S_Images")

with open(ann_file, "r") as f:
    lines = f.readlines()
random.shuffle(lines)

n_total = len(lines)
n_val = int(SPLITS_INFO["val"] * n_total)
split_dict = {"val": lines[:n_val], "test": lines[n_val:]}

for split, split_lines in split_dict.items():
    for line in split_lines:
        parts = line.strip().split(",")
        if len(parts) < 5:
            skipped_files.append(line.strip())
            continue
        filename = parts[0]
        xmin, xmax, ymin, ymax = map(float, parts[1:5])

        img_path = os.path.join(img_dir, filename)
        if not os.path.exists(img_path):
            skipped_files.append(filename)
            continue

        img_w, img_h = Image.open(img_path).size
        x_center = max(0.0, min(1.0, (xmin + xmax) / 2 / img_w))
        y_center = max(0.0, min(1.0, (ymin + ymax) / 2 / img_h))
        w = max(0.0, min(1.0, (xmax - xmin) / img_w))
        h = max(0.0, min(1.0, (ymax - ymin) / img_h))

        dst_filename = get_unique_filename()
        dst_img = Path(TARGET) / "images" / split / dst_filename
        shutil.copy(img_path, dst_img)

        dst_label = Path(TARGET) / "labels" / split / (Path(dst_filename).stem + ".txt")
        with open(dst_label, "a") as f:
            f.write(f"{CLASS_ID} {x_center} {y_center} {w} {h}\n")

print("COCO Hand-S Done")

# ========== Step 3: COCO Hand-Big ==========
ann_file = os.path.join(
    COCO_HAND_ROOT, "COCO-Hand-Big", "COCO-Hand-Big_annotations.txt"
)
img_dir = os.path.join(COCO_HAND_ROOT, "COCO-Hand-Big", "COCO-Hand-Big_Images")

with open(ann_file, "r") as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split(",")
    if len(parts) < 5:
        skipped_files.append(line.strip())
        continue
    filename = parts[0]
    xmin, xmax, ymin, ymax = map(float, parts[1:5])

    img_path = os.path.join(img_dir, filename)
    if not os.path.exists(img_path):
        skipped_files.append(filename)
        continue

    img_w, img_h = Image.open(img_path).size
    x_center = max(0.0, min(1.0, (xmin + xmax) / 2 / img_w))
    y_center = max(0.0, min(1.0, (ymin + ymax) / 2 / img_h))
    w = max(0.0, min(1.0, (xmax - xmin) / img_w))
    h = max(0.0, min(1.0, (ymax - ymin) / img_h))

    dst_filename = get_unique_filename()
    dst_img = Path(TARGET) / "images" / "train" / dst_filename
    shutil.copy(img_path, dst_img)

    dst_label = Path(TARGET) / "labels" / "train" / (Path(dst_filename).stem + ".txt")
    with open(dst_label, "a") as f:
        f.write(f"{CLASS_ID} {x_center} {y_center} {w} {h}\n")

print("COCO Hand-Big Done")

# ========== Step 4: Oxford Hand ==========
for split in ["training_dataset", "validation_dataset", "test_dataset"]:
    if "training" in split:
        split_name = "train"
    elif "validation" in split:
        split_name = "val"
    else:
        split_name = "test"

    ann_dir = os.path.join(
        OXFORD_HAND_ROOT, split, split.replace("_dataset", "_data"), "annotations"
    )
    img_dir = os.path.join(
        OXFORD_HAND_ROOT, split, split.replace("_dataset", "_data"), "images"
    )

    for ann_file in os.listdir(ann_dir):
        if not ann_file.endswith(".mat"):
            continue

        stem = Path(ann_file).stem
        img_file = stem + ".jpg"
        img_path = os.path.join(img_dir, img_file)
        if not os.path.exists(img_path):
            skipped_files.append(img_file)
            continue

        # ====== 三次重试机制 ======
        for attempt in range(MAX_RETRIES):
            try:
                mat = sio.loadmat(os.path.join(ann_dir, ann_file), simplify_cells=True)
                if "boxes" not in mat or len(mat["boxes"]) == 0:
                    raise ValueError("boxes 为空或不存在")
                boxes = mat["boxes"]
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    skipped_files.append(
                        f"{os.path.join(ann_dir, ann_file)} ({str(e)})"
                    )
                    boxes = []

        if len(boxes) == 0:
            continue
        if isinstance(boxes, dict):
            boxes = [boxes]

        img_w, img_h = Image.open(img_path).size
        bboxes = []

        for hand in boxes:
            points = [hand["a"], hand["b"], hand["c"], hand["d"]]
            x_coords = [p[1] for p in points]
            y_coords = [p[0] for p in points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            x_center = max(0.0, min(1.0, (xmin + xmax) / 2 / img_w))
            y_center = max(0.0, min(1.0, (ymin + ymax) / 2 / img_h))
            w = max(0.0, min(1.0, (xmax - xmin) / img_w))
            h = max(0.0, min(1.0, (ymax - ymin) / img_h))

            bboxes.append([CLASS_ID, x_center, y_center, w, h])

        dst_filename = get_unique_filename()
        dst_img = Path(TARGET) / "images" / split_name / dst_filename
        shutil.copy(img_path, dst_img)

        dst_label = (
            Path(TARGET) / "labels" / split_name / (Path(dst_filename).stem + ".txt")
        )
        with open(dst_label, "w") as f:
            for box in bboxes:
                f.write(" ".join(map(str, box)) + "\n")

print("Oxford Hand Done.")

# ========== Step 5: 打印结果 ==========
print("✅ 数据集已全部转换为 YOLO 格式:", TARGET)
print(f"⚠️ 跳过的文件总数: {len(skipped_files)}")
print("前 10 个被跳过的文件:")
for name in skipped_files[:10]:
    print("   ", name)

yaml_path = Path(TARGET) / "hands.yaml"
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(
        f"""path: {Path(TARGET).resolve()}
train: images/train
val: images/val
test: images/test

nc: 1
names: ["hand"]
"""
    )

print(f"✅ 已生成配置文件: {yaml_path}")
