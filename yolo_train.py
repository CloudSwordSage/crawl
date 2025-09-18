# -*- coding: utf-8 -*-
# @Time    : 2025/8/31 09:07:43
# @Author  : 墨烟行(GitHub UserName: CloudSwordSage)
# @File    : yolo_train.py
# @Desc    : YOLO 训练文件

from ultralytics import YOLO
import multiprocessing as mp


def main():
    model = YOLO("./Hand/train2/weights/best.pt")
    model.train(
        data="./datasets/Hands/hands.yaml",
        epochs=400,
        batch=-1,
        save=True,
        plots=True,
        project="./Hand",
        cos_lr=True,
        augment=True,
    )


if __name__ == "__main__":
    mp.set_start_method("spawn")
    mp.freeze_support()  # Python 多进程不加会抛出警告
    main()
