# -*- coding: utf-8 -*-
# @Time    : 2025/9/18 13:27:38
# @Author  : 墨烟行(GitHub UserName: CloudSwordSage)
# @File    : test.py
# @Desc    : 手部关键点检测测试脚本

from ultralytics import YOLO
import cv2
import numpy as np
from util import get_keypoints, oks, oks_nms, HandKalmanFilter
from config import HRNetConfig
from HRNet import HRNet
import torch

from cv2.ximgproc import guidedFilter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_inside(box1, box2):
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    return x1 >= a1 and y1 >= b1 and x2 <= a2 and y2 <= b2


def filter_boxes(boxes, confidences):
    sorted_indices = np.argsort(confidences)[::-1]
    sorted_boxes = [boxes[i] for i in sorted_indices]
    sorted_confs = [confidences[i] for i in sorted_indices]

    filtered_boxes = []
    filtered_confs = []

    for i in range(len(sorted_boxes)):
        current_box = sorted_boxes[i]
        current_conf = sorted_confs[i]
        keep = True

        for j in range(len(filtered_boxes)):
            if is_inside(current_box, filtered_boxes[j]):
                keep = False
                break

        if keep:
            filtered_boxes.append(current_box)
            filtered_confs.append(current_conf)

    return filtered_boxes, filtered_confs


def draw_boxes(frame, boxes, confidences, classes):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        conf = confidences[i]
        cls = classes[i]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"Class {cls}: {conf:.2f}"
        cv2.putText(
            frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    return frame


def crop_and_resize_hand(frame, box, target_size=(256, 192), expand_pixels=20):
    """根据YOLO检测框裁剪和缩放手部区域，可向外扩展像素"""
    x1, y1, x2, y2 = map(int, box)

    # 获取图像尺寸
    img_h, img_w = frame.shape[:2]

    # 向外扩展检测框
    x1 = max(0, x1 - expand_pixels)
    y1 = max(0, y1 - expand_pixels)
    x2 = min(img_w, x2 + expand_pixels)
    y2 = min(img_h, y2 + expand_pixels)

    # 裁剪手部区域
    hand_crop = frame[y1:y2, x1:x2]
    if hand_crop.size == 0:
        return None, None

    # 获取原始尺寸
    orig_h, orig_w = hand_crop.shape[:2]
    target_h, target_w = target_size

    # 计算缩放比例
    scale_h = target_h / orig_h
    scale_w = target_w / orig_w
    scale = min(scale_h, scale_w)

    # 缩放图像
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)
    resized = cv2.resize(hand_crop, (new_w, new_h))

    # 创建目标尺寸的空白图像
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # 计算填充位置
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2

    # 将缩放后的图像放入中心位置
    padded[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

    # 返回处理后的图像和变换参数
    transform_params = {
        "orig_box": (x1, y1, x2, y2),
        "scale": scale,
        "pad_top": pad_top,
        "pad_left": pad_left,
        "target_size": target_size,
    }

    return padded, transform_params


def transform_keypoints_to_original(keypoints_norm, transform_params):
    """将归一化关键点坐标转换回原始图像坐标系"""
    target_h, target_w = transform_params["target_size"]
    scale = transform_params["scale"]
    pad_top = transform_params["pad_top"]
    pad_left = transform_params["pad_left"]
    x1, y1, x2, y2 = transform_params["orig_box"]

    # 将归一化坐标转换为裁剪图像中的像素坐标
    keypoints_pixel = keypoints_norm * np.array([target_w, target_h])

    # 减去填充并除以缩放比例
    keypoints_crop = np.zeros_like(keypoints_pixel)
    keypoints_crop[:, 0] = (keypoints_pixel[:, 0] - pad_left) / scale
    keypoints_crop[:, 1] = (keypoints_pixel[:, 1] - pad_top) / scale

    # 转换回原始图像坐标系
    keypoints_original = np.zeros_like(keypoints_crop)
    keypoints_original[:, 0] = keypoints_crop[:, 0] + x1
    keypoints_original[:, 1] = keypoints_crop[:, 1] + y1

    return keypoints_original


def draw_keypoints(frame, keypoints, connections=None, confidence_threshold=0.3):
    """在图像上绘制关键点和连接线"""
    if connections is None:
        # 手部关键点连接关系
        # fmt: off
        connections = [
            [0, 1], [1, 2], [2, 3], [3, 4],  # 拇指
            [0, 5], [5, 6], [6, 7], [7, 8],  # 食指
            [0, 9], [9, 10], [10, 11], [11, 12],  # 中指
            [0, 13], [13, 14], [14, 15], [15, 16],  # 无名指
            [0, 17], [17, 18], [18, 19], [19, 20],  # 小指
        ]
        # fmt: on

    # 绘制关键点
    for i, (x, y) in enumerate(keypoints):
        if x > 0 and y > 0:  # 只绘制有效关键点
            cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
            cv2.putText(
                frame,
                str(i),
                (int(x) + 5, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
            )

    # 绘制连接线
    for connection in connections:
        p1, p2 = connection
        x1, y1 = keypoints[p1]
        x2, y2 = keypoints[p2]
        if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:  # 只连接有效关键点
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    return frame


def preprocess_image_for_hrnet(image):
    """预处理图像供HRNet使用"""
    # 转换为RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_normalized = (image_rgb.astype(np.float32) / 255.0 - mean) / std

    # 转换为Tensor并添加批次维度
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).float()
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


# 加载模型
model = YOLO("./Hand/train3/weights/best.pt")
hr_net = HRNet(HRNetConfig())
hr_net.load_state_dict(torch.load("./checkpoints/best_model.pth"))
hr_net.eval().to(device)
hand_kf = HandKalmanFilter(21)
# 打开摄像头
cap = cv2.VideoCapture(0)

# 阈值和上一个满足条件的点集合
kp_threshold = 0.1  # KP Score阈值
last_valid_keypoints = None  # 保存上一个满足条件的点集合


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_denoised = guidedFilter(
        guide=frame.copy(), src=frame.copy(), radius=1, eps=500
    )
    frame_denoised = cv2.GaussianBlur(frame_denoised, (5, 5), 0)

    # YOLO检测手部
    results = model(frame_denoised, conf=0.5, verbose=False)

    boxes = []
    confidences = []
    classes = []

    for result in results:
        for box in result.boxes:
            bbox = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = box.cls[0].item()

            boxes.append(bbox)
            confidences.append(conf)
            classes.append(cls)

    # 过滤重叠框
    filtered_boxes, filtered_confs = filter_boxes(boxes, confidences)

    filtered_classes = []
    for i in range(len(filtered_boxes)):
        original_idx = boxes.index(filtered_boxes[i])
        filtered_classes.append(classes[original_idx])

    # 绘制检测框
    annotated_frame = draw_boxes(
        frame.copy(), filtered_boxes, filtered_confs, filtered_classes
    )

    # annotated_frame = frame_denoised.copy()

    # 对每个检测到的手部进行处理
    drawing_data = []
    kp_list = []
    box_scores = []
    box_areas = []

    for i, box in enumerate(filtered_boxes):
        # 裁剪和缩放手部区域
        hand_image, transform_params = crop_and_resize_hand(
            frame_denoised,
            box,
            expand_pixels=int(max(box[3] - box[1], box[2] - box[0]) // 2),
        )
        if hand_image is None:
            continue

        median_filtered = cv2.medianBlur(hand_image, 3)
        bilateral_filtered = cv2.bilateralFilter(
            median_filtered, d=5, sigmaColor=75, sigmaSpace=75
        )

        # 预处理图像供HRNet使用
        input_tensor = preprocess_image_for_hrnet(bilateral_filtered)
        input_tensor = input_tensor.to(device)

        # 使用HRNet进行推理
        with torch.no_grad():
            heatmaps = hr_net(input_tensor)

        # 获取关键点
        heatmaps_np = heatmaps.cpu().numpy()[0]  # 移除批次维度
        keypoints_norm, scores = get_keypoints(heatmaps_np, window_size=5)

        # 将关键点转换回原始图像坐标系
        keypoints_original = transform_keypoints_to_original(
            keypoints_norm, transform_params
        )

        # 计算平均置信度
        avg_score = np.mean(scores)

        # 判断是否使用当前关键点还是缓存的关键点
        if avg_score >= kp_threshold:
            points_to_draw = keypoints_original
            last_valid_keypoints = keypoints_original.copy()
            status_text = f"KP Score: {avg_score:.2f} (Real-time)"
        elif last_valid_keypoints is not None:
            points_to_draw = last_valid_keypoints
            status_text = f"KP Score: {avg_score:.2f} (Using cached)"
        else:
            points_to_draw = keypoints_original
            status_text = f"KP Score: {avg_score:.2f} (Low confidence)"

        # points_to_draw = hand_kf.update(points_to_draw)
        # alpha = 0.6  # 平滑系数
        # if last_valid_keypoints is None:
        #     smoothed = keypoints_original
        # else:
        #     smoothed = alpha * keypoints_original + (1 - alpha) * last_valid_keypoints
        # last_valid_keypoints = smoothed
        # points_to_draw = smoothed

        # 收集用于OKS-NMS的数据
        kp_list.append(points_to_draw)
        box_scores.append(filtered_confs[i])
        box_areas.append((box[2] - box[0]) * (box[3] - box[1]))

        x1, y1, x2, y2 = map(int, box)
        drawing_data.append(
            {
                "points": points_to_draw,
                "status_text": status_text,
                "box_coords": (x1, y1, x2, y2),
            }
        )

    # ========= 新增：OKS-NMS =========
    if len(kp_list) > 1:
        kp_list = np.stack(kp_list)
        box_scores = np.array(box_scores)
        box_areas = np.array(box_areas)

        keep_idx = oks_nms(
            filtered_boxes, kp_list, box_scores, box_areas, oks_threshold=0.7
        )
        drawing_data = [drawing_data[i] for i in keep_idx]

    # 绘制
    for data in drawing_data:
        annotated_frame = draw_keypoints(annotated_frame, data["points"])

        x1, y1, x2, y2 = data["box_coords"]
        cv2.putText(
            annotated_frame,
            data["status_text"],
            (x1, y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    cv2.imshow("Hand Keypoints Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
