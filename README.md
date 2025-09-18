# CRAWL

一个基于HRNet架构的实时手部21个关键点检测系统，结合YOLO手部检测和HRNet关键点定位。

<div align="center">

[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](https://swanlab.cn/@CloudSwordSage/HRNet/overview)

</div>

## 项目概述

本项目实现了一个端到端的手部关键点检测系统，使用YOLO进行手部区域检测，然后使用HRNet进行精确的21个关键点定位。系统支持实时摄像头输入和视频处理。

## 主要特性

- 🎯 **高精度检测**: 使用HRNet架构实现21个手部关键点的高精度定位
- ⚡ **实时性能**: 支持实时摄像头输入处理
- 🔍 **双重检测**: YOLO手部检测 + HRNet关键点定位
- 📊 **性能监控**: 集成SwanLab进行训练监控和指标可视化
- 🛡️ **鲁棒性**: 包含OKS-NMS后处理技术
- 🎮 **交互式演示**: 实时显示检测结果和置信度信息

## 技术架构

### 模型结构
- **YOLOv8**: 用于手部区域检测
- **HRNet**: 用于手部关键点热图预测
- **后处理**: OKS-NMS、关键点坐标转换

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖库:
- torch==2.6.0+cu124
- torchvision==0.21.0+cu124
- ultralytics==8.3.158
- opencv-python
- numpy
- swanlab==0.6.10 (训练监控)

## 项目结构

```
.
├── config.py          # 模型和训练配置
├── data.py           # 数据加载和预处理
├── HRNet.py          # HRNet模型定义
├── loss.py           # 损失函数定义
├── train.py          # 训练脚本
├── test.py           # 实时测试脚本
├── util.py           # 工具函数
├── yolo_train.py     # YOLO训练脚本
├── freiHand2coco.py  # 数据集格式转换
├── hand2yolo.py      # YOLO格式转换
├── requirements.txt  # 依赖列表
├── checkpoints/      # 模型检查点
├── datasets/         # 数据集目录
├── Hand/            # YOLO训练相关文件
└── background/       # 背景图像
```

## 快速开始

### 1. 数据准备

确保数据集位于 `datasets/` 目录下，支持FreiHand等标准手部数据集格式。

### 2. 训练模型

**训练HRNet关键点检测模型:**
```bash
python train.py
```

**训练YOLO手部检测模型:**
```bash
python yolo_train.py
```

### 3. 实时测试

运行实时摄像头演示:
```bash
python test.py
```

## 使用方法

### 训练配置

在 `config.py` 中调整训练参数:

```python
# 调整训练参数
train_config = TrainConfig(
    epochs=50,
    lr=1e-2,
    batch_size=16,
    img_size=[256, 192]
)
```

### 模型推理

使用训练好的模型进行推理:

```python
from test import preprocess_image_for_hrnet, draw_keypoints

# 加载模型
model = HRNet(HRNetConfig())
model.load_state_dict(torch.load("./checkpoints/best_model.pth"))

# 预处理图像
input_tensor = preprocess_image_for_hrnet(image)

# 推理
with torch.no_grad():
    heatmaps = model(input_tensor)

# 后处理和可视化
keypoints = get_keypoints(heatmaps)
result_image = draw_keypoints(image, keypoints)
```

## 性能指标

系统使用以下指标进行评估:
- **mAP@50**: 50% IoU阈值下的平均精度
- **mAP@50:95**: 50%-95% IoU阈值下的平均精度
- **OKS**: Object Keypoint Similarity
- **推理速度**: 实时处理性能

## 数据集

### 所需数据集

本项目需要以下数据集进行训练：

1. **FreiHand数据集** - 用于HRNet关键点检测训练
2. **COCO-Hand数据集** - 用于YOLO手部检测训练
3. **Oxford Hand数据集** - 用于YOLO手部检测训练

### 数据格式转换

使用提供的转换脚本进行格式转换:

```bash
# 转换FreiHand到COCO关键点格式
python freiHand2coco.py

# 转换COCO-Hand和Oxford Hand到YOLO格式
python hand2yolo.py
```

数据集应放置在 `datasets/` 目录下，并按照转换脚本的要求组织文件结构。

## 许可证

本项目基于 APACHE2.0 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 致谢

- HRNet 原论文作者
- Ultralytics YOLO 团队
- FreiHand 等数据集提供者

## 联系方式

- 作者: 墨烟行 (CloudSwordSage)
- GitHub: [https://github.com/CloudSwordSage](https://github.com/CloudSwordSage)

## 更新日志

### v1.0.0 (2025-09-18)
- 初始版本发布
- 支持实时手部关键点检测
- 集成HRNet和YOLO模型
- 添加训练和评估脚本

---

**注意**: 确保在运行前安装所有依赖，并准备好相应的数据集。对于实时演示，需要可用的摄像头设备。
