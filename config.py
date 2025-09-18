# -*- coding: utf-8 -*-
# @Time    : 2025/8/31 10:19:56
# @Author  : 墨烟行(GitHub UserName: CloudSwordSage)
# @File    : config.py
# @Desc    : 配置文件

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class StageConfig:
    num_modules: int
    num_branches: int
    num_blocks: List[int]
    num_channels: List[int]
    block: str = "BOTTLENECK"
    fuse_method: str = "SUM"


@dataclass
class ModelConfig:
    num_joints: int = 21
    input_height: int = 256
    input_width: int = 192
    pretrained: str = ""
    final_conv_kernel: int = 1


@dataclass
class HRNetConfig:
    stage2: StageConfig = field(
        default_factory=lambda: StageConfig(
            num_modules=1,
            num_branches=2,
            num_blocks=[4, 4],
            num_channels=[32, 64],
            block="BASIC",
        )
    )
    stage3: StageConfig = field(
        default_factory=lambda: StageConfig(
            num_modules=4,
            num_branches=3,
            num_blocks=[4, 4, 4],
            num_channels=[32, 64, 128],
            block="BASIC",
        )
    )
    stage4: StageConfig = field(
        default_factory=lambda: StageConfig(
            num_modules=3,
            num_branches=4,
            num_blocks=[4, 4, 4, 4],
            num_channels=[32, 64, 128, 256],
            block="BASIC",
        )
    )
    model: ModelConfig = field(default_factory=lambda: ModelConfig(num_joints=21))


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 1e-2
    batch_size: int = 16
    num_workers: int = 4
    prefetch_factor: int = 2
    use_augmentation: bool = True
    img_size: List[int] = field(default_factory=lambda: [256, 192])
    momentum: float = 0.9
    weight_decay: float = 1e-4
    eta_min: float = 0.0001
    early_stop_patience: int = 30

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "prefetch_factor": self.prefetch_factor,
            "use_augmentation": self.use_augmentation,
            "img_size": self.img_size,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "eta_min": self.eta_min,
            "early_stop_patience": self.early_stop_patience,
        }
