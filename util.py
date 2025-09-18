# -*- coding: utf-8 -*-
# @Time    : 2025/9/18 16:10:41
# @Author  : 墨烟行(GitHub UserName: CloudSwordSage)
# @File    : util.py
# @Desc    : 

# -*- coding: utf-8 -*-
# @Time    : 2025/8/31 14:44:41
# @Author  : 墨烟行(GitHub UserName: CloudSwordSage)
# @File    : util.py
# @Desc    : 手部关键点工具函数

import numpy as np
import cv2


def oks(g, d, a_g=1.0, sigmas=None, in_vis_thre=None):
    """
    g: [K,2] GT关键点
    d: [K,2] 预测关键点
    """
    if sigmas is None:
        # fmt: off
        sigmas = np.array([
            .26, .25, .25, .35, .35,
            .79, .79, .72, .72, .62,
            .62, 1.07, 1.07, .87, .87,
            .89, .89, .89, .89, .89, .89
        ]) / 10.0
        # fmt: on

    vars = (sigmas * 2) ** 2
    n_joints = g.shape[0]

    dx = d[:, 0] - g[:, 0]
    dy = d[:, 1] - g[:, 1]
    e = (dx**2 + dy**2) / vars / a_g / 2

    if in_vis_thre is not None:
        mask = np.ones(n_joints, dtype=bool)
        e = e[mask]

    return np.sum(np.exp(-e)) / n_joints if n_joints > 0 else 0.0


def AP(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def mAP(gts, dts, scores, oks_thrs, a_g=1.0, sigmas=None, in_vis_thre=None):
    """
    gts: list of [K,2]
    dts: list of [K,2]
    scores: list of [K,] or [1,]
    """
    aps = []

    for thr in oks_thrs:
        tp, fp = [], []

        for gt, dt, sc in zip(gts, dts, scores):
            o = oks(gt, dt, a_g=a_g, sigmas=sigmas, in_vis_thre=in_vis_thre)
            if o >= thr:
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        rec = tp / len(gts)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        aps.append(AP(rec, prec))

    return np.mean(aps)


def mAP50(gts, dts, scores, a_g=1.0, sigmas=None, in_vis_thre=None):
    return mAP(
        gts,
        dts,
        scores,
        oks_thrs=[0.5],
        a_g=a_g,
        sigmas=sigmas,
        in_vis_thre=in_vis_thre,
    )


def mAP50_95(gts, dts, scores, a_g=1.0, sigmas=None, in_vis_thre=None):
    oks_thrs = np.arange(0.5, 1.0, 0.05)
    return mAP(
        gts,
        dts,
        scores,
        oks_thrs=oks_thrs,
        a_g=a_g,
        sigmas=sigmas,
        in_vis_thre=in_vis_thre,
    )


def get_keypoints(heatmaps, window_size=3, eps=1e-6):
    num_joints, H, W = heatmaps.shape
    coords = np.zeros((num_joints, 2), dtype=np.float32)
    maxvals = np.zeros((num_joints, 1), dtype=np.float32)

    sigma = window_size / 2
    ax = np.arange(-window_size // 2 + 1, window_size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    gaussian = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    for i in range(num_joints):
        hm = heatmaps[i]
        idx = np.argmax(hm)
        y, x = divmod(idx, W)
        maxvals[i] = hm[y, x]

        x0, y0 = max(0, x - window_size // 2), max(0, y - window_size // 2)
        x1, y1 = min(W, x + window_size // 2 + 1), min(H, y + window_size // 2 + 1)

        patch = hm[y0:y1, x0:x1]
        g_patch = gaussian[
            (y0 - y + window_size // 2) : (y1 - y + window_size // 2),
            (x0 - x + window_size // 2) : (x1 - x + window_size // 2),
        ]

        weighted = patch * g_patch
        total = np.sum(weighted) + eps

        yy_idx, xx_idx = np.mgrid[y0:y1, x0:x1]
        cx = np.sum(weighted * xx_idx) / total
        cy = np.sum(weighted * yy_idx) / total

        coords[i] = [cx, cy]

    scale_x = 1.0 / W
    scale_y = 1.0 / H
    coords[:, 0] *= scale_x
    coords[:, 1] *= scale_y

    return coords, maxvals


def oks_nms(boxes, keypoints_list, scores, areas, oks_threshold=0.9):
    """
    基于 OKS 的 NMS
    boxes: N x 4
    keypoints_list: N x K x 2 (像素坐标)
    scores: N (YOLO置信度)
    areas: N (每个手部检测框面积或全图面积)
    """
    keep = []
    idxs = np.argsort(scores)[::-1]  # 按置信度从高到低排序

    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)

        oks_list = []
        for j in idxs[1:]:
            oks_val = oks(
                keypoints_list[i],
                keypoints_list[j],
                a_g=max(areas[i], areas[j]),  # 用较大面积做归一化
            )
            oks_list.append(oks_val)

        oks_list = np.array(oks_list)
        # 保留 OKS 小于阈值的框
        idxs = idxs[1:][oks_list < oks_threshold]

    return keep


class HandKalmanFilter:
    def __init__(self, num_keypoints):
        """
        num_keypoints: 每只手的关键点数量
        """
        self.num_keypoints = num_keypoints
        self.filters = [cv2.KalmanFilter(4, 2) for _ in range(num_keypoints)]
        for kf in self.filters:
            # 状态向量 [x, y, dx, dy], 测量 [x, y]
            kf.transitionMatrix = np.array(
                [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
            )
            kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
            kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
            kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
            kf.errorCovPost = np.eye(4, dtype=np.float32)
            kf.statePost = np.zeros((4, 1), np.float32)

    def update(self, keypoints):
        """
        keypoints: np.array of shape (num_keypoints, 2)
        返回滤波后的关键点 np.array (num_keypoints, 2)
        """
        smoothed = []
        for i, kf in enumerate(self.filters):
            measurement = np.array(
                [[np.float32(keypoints[i, 0])], [np.float32(keypoints[i, 1])]]
            )
            kf.correct(measurement)
            pred = kf.predict()
            smoothed.append([pred[0, 0], pred[1, 0]])
        return np.array(smoothed)


if __name__ == "__main__":
    np.random.seed(42)
    num_joints = 21

    # 构造 GT
    g = np.random.uniform(0.3, 0.7, size=(num_joints, 2))

    # 构造预测
    noise = np.random.normal(0, 0.05, size=(num_joints, 2))
    d = g + noise
    scores = np.ones(1)  # 单目标置信度为1

    print("GT shape:", g.shape)
    print("Pred shape:", d.shape)
    print("Scores:", scores)

    # OKS
    oks_val = oks(g, d)
    print("OKS:", oks_val)

    # mAP
    gts_list = [g]
    dts_list = [d]
    scores_list = [scores]
    print("mAP@50:", mAP50(gts_list, dts_list, scores_list))
    print("mAP@0.5:0.95:", mAP50_95(gts_list, dts_list, scores_list))

    # 测试 get_keypoints
    heatmaps = np.random.rand(num_joints, 64, 48)
    coords, maxvals = get_keypoints(heatmaps, window_size=5)
    print("Keypoints shape:", coords.shape)
    print("Maxvals shape:", maxvals.shape)
