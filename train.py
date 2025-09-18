# -*- coding: utf-8 -*-
# @Time    : 2025/9/2 14:33:03
# @Author  : 墨烟行(GitHub UserName: CloudSwordSage)
# @File    : train.py
# @Desc    : 训练循环

import os

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from tqdm.auto import tqdm

from util import mAP50, mAP50_95, mAP, oks, get_keypoints
from data import create_data_loaders
from HRNet import HRNet, print_model
from loss import JointsMSELoss

from config import HRNetConfig, TrainConfig

import swanlab

if __name__ == "__main__":
    net_config = HRNetConfig()
    train_config = TrainConfig()
    print(train_config.to_dict())

    run = swanlab.init(project="HRNet", config=train_config.to_dict())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = create_data_loaders(
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        prefetch_factor=train_config.prefetch_factor,
        image_size=train_config.img_size,
        use_augmentation=train_config.use_augmentation,
    )

    net = HRNet(net_config).to(device)
    print_model(net)
    criterion = JointsMSELoss().to(device)
    optimizer = optim.SGD(
        net.parameters(),
        lr=train_config.lr,
        momentum=train_config.momentum,
        weight_decay=train_config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_config.epochs - 10, eta_min=train_config.eta_min
    )

    best_map50 = 0.0
    patience = train_config.early_stop_patience
    counter = 0
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")
    last_model_path = os.path.join(save_dir, "last_model.pth")

    for epoch in range(train_config.epochs):
        net.train()
        train_loss = 0.0

        loop = tqdm(
            train_loader, desc=f"Epoch [{epoch+1}/{train_config.epochs}] Training"
        )
        for batch in loop:
            images = batch["image"].to(device)
            heatmaps_gt = batch["heatmaps"].to(device)
            keypoints_gt = batch["keypoints"].to(device)
            areas = batch["areas"].to(device)

            optimizer.zero_grad()
            outputs = net(images)

            loss = criterion(outputs, heatmaps_gt)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}")
        run.log({"train_loss": train_loss, "epoch": epoch + 1})

        net.eval()
        val_loss = 0.0
        oks_scores_all = []
        gts_list_all = []
        dts_list_all = []
        scores_list_all = []
        loop = tqdm(
            val_loader, desc=f"Epoch [{epoch+1}/{train_config.epochs}] Validation"
        )

        with torch.no_grad():
            for batch in loop:
                images = batch["image"].to(device)
                heatmaps_gt = batch["heatmaps"].to(device)
                keypoints_gt = batch["keypoints"].to(device)
                areas = batch["areas"].to(device)

                outputs = net(images)
                loss = criterion(outputs, heatmaps_gt)
                val_loss += loss.item() * images.size(0)

                B, K, H_hm, W_hm = outputs.shape
                coords_pred_list = []
                scores_list = []

                for b in range(B):
                    coords_pred, scores = get_keypoints(
                        outputs[b].cpu().numpy(), window_size=5
                    )
                    coords_pred_list.append(coords_pred)
                    scores_list.append(scores.squeeze(-1))

                coords_pred = np.stack(coords_pred_list)
                scores_pred = np.stack(scores_list)

                keypoints_gt_np = np.stack(
                    [
                        k[:, :2].cpu().numpy() if hasattr(k, "cpu") else k[:, :2]
                        for k in keypoints_gt
                    ]
                )

                img_h, img_w = train_config.img_size
                keypoints_gt_pix = keypoints_gt_np * np.array([img_w, img_h])
                coords_pred_pix = coords_pred * np.array([img_w, img_h])

                for b in range(B):
                    oks_val = oks(
                        keypoints_gt_pix[b], coords_pred_pix[b], a_g=areas[b].item()
                    )
                    oks_scores_all.append(oks_val)

                for b in range(B):
                    gts_list_all.append(keypoints_gt_pix[b])
                    dts_list_all.append(coords_pred_pix[b])
                    scores_list_all.append(scores_pred[b])

        val_loss /= len(val_loader.dataset)
        mean_oks = np.mean(oks_scores_all)
        map50 = mAP50(gts_list_all, dts_list_all, scores_list_all, a_g=areas[b].item())
        map50_95 = mAP50_95(
            gts_list_all, dts_list_all, scores_list_all, a_g=areas[b].item()
        )
        map10 = mAP(
            gts_list_all,
            dts_list_all,
            scores_list_all,
            oks_thrs=[0.1],
            a_g=areas[b].item(),
        )

        print(
            f"Epoch {epoch+1}: Val Loss = {val_loss:.6f}, OKS = {mean_oks:.4f}, mAP@50 = {map50:.4f}, mAP@50:95 = {map50_95:.4f}, mAP@10 = {map10:.4f}"
        )
        run.log(
            {
                "val_loss": val_loss,
                "val_oks": mean_oks,
                "map50": map50,
                "map50_95": map50_95,
                "epoch": epoch + 1,
            }
        )
        torch.save(net.state_dict(), last_model_path)
        if map50 > 0.0:
            if map50 > best_map50:
                best_map50 = map50
                counter = 0
                torch.save(net.state_dict(), best_model_path)
                print(
                    f"New best model saved at epoch {epoch+1} with mAP@50 = {best_map50:.4f}"
                )
            else:
                counter += 1
                if counter >= patience:
                    print(
                        f"Early stopping triggered at epoch {epoch+1}. Best mAP@50 = {best_map50:.4f}"
                    )
                    break
