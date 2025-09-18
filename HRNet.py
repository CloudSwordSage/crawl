# -*- coding: utf-8 -*-
# @Time    : 2025/8/31 09:49:23
# @Author  : 墨烟行(GitHub UserName: CloudSwordSage)
# @File    : HRNet.py
# @Desc    : HRNet 模型定义

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out)
        return out


blocks_dict = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck}


class HRModule(nn.Module):
    def __init__(
        self,
        num_branches,
        block,
        num_blocks: List[int],
        num_inchannels: List[int],
        num_channels: List[int],
        fuse_method="SUM",
        multi_scale_output=True,
    ):
        super().__init__()
        self.num_branches = num_branches
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.multi_scale_output = multi_scale_output

        self.branches = nn.ModuleList(
            [
                self._make_one_branch(i, block, num_blocks, num_channels)
                for i in range(num_branches)
            ]
        )

        self.fuse_layers = self._make_fuse_layers()

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels):
        layers = []
        inplanes = self.num_inchannels[branch_index]
        outplanes = num_channels[branch_index] * block.expansion

        downsample = None
        if inplanes != outplanes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, 1, bias=False), nn.BatchNorm2d(outplanes)
            )

        layers.append(
            block(inplanes, num_channels[branch_index], downsample=downsample)
        )
        inplanes = outplanes
        for _ in range(1, num_blocks[branch_index]):
            layers.append(block(inplanes, num_channels[branch_index]))
        self.num_inchannels[branch_index] = outplanes
        return nn.Sequential(*layers)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        fuse_layers = nn.ModuleList()
        for i in range(self.num_branches if self.multi_scale_output else 1):
            fuse_layer = nn.ModuleList()
            for j in range(self.num_branches):
                if i == j:
                    fuse_layer.append(None)
                elif j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                self.num_inchannels[j],
                                self.num_inchannels[i],
                                1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(self.num_inchannels[i]),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                        )
                    )
                else:
                    convs = []
                    for k in range(i - j):
                        inch = (
                            self.num_inchannels[j]
                            if k != i - j - 1
                            else self.num_inchannels[i]
                        )
                        convs.append(
                            nn.Sequential(
                                nn.Conv2d(
                                    self.num_inchannels[j], inch, 3, 2, 1, bias=False
                                ),
                                nn.BatchNorm2d(inch),
                                (
                                    nn.ReLU(inplace=True)
                                    if k != i - j - 1
                                    else nn.Identity()
                                ),
                            )
                        )
                    fuse_layer.append(nn.Sequential(*convs))
            fuse_layers.append(fuse_layer)
        return fuse_layers

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        x = [branch(xi) for branch, xi in zip(self.branches, x)]
        if self.num_branches == 1:
            return x
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(F.relu(y))
        return x_fuse


class HRNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.inplanes = 64
        self.cfg = cfg

        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        num_channels = [
            c * blocks_dict[cfg.stage2.block].expansion for c in cfg.stage2.num_channels
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(cfg.stage2, num_channels)

        num_channels = [
            c * blocks_dict[cfg.stage3.block].expansion for c in cfg.stage3.num_channels
        ]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(cfg.stage3, num_channels)

        num_channels = [
            c * blocks_dict[cfg.stage4.block].expansion for c in cfg.stage4.num_channels
        ]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            cfg.stage4, num_channels, multi_scale_output=False
        )

        self.final_layer = nn.Conv2d(
            pre_stage_channels[0],
            cfg.model.num_joints,
            kernel_size=cfg.model.final_conv_kernel,
            stride=1,
            padding=1 if cfg.model.final_conv_kernel == 3 else 0,
        )

    def _make_layer(self, block, planes, blocks):
        layers = []
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion, 1, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )
        layers.append(block(self.inplanes, planes, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_channels_pre, num_channels_cur):
        layers = []
        for i, cur_c in enumerate(num_channels_cur):
            if i < len(num_channels_pre):
                if cur_c != num_channels_pre[i]:
                    layers.append(
                        nn.Sequential(
                            nn.Conv2d(num_channels_pre[i], cur_c, 3, 1, 1, bias=False),
                            nn.BatchNorm2d(cur_c),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    layers.append(None)
            else:
                convs = []
                inch = num_channels_pre[-1]
                for j in range(i + 1 - len(num_channels_pre)):
                    outc = cur_c if j == i - len(num_channels_pre) else inch
                    convs.append(
                        nn.Sequential(
                            nn.Conv2d(inch, outc, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(outc),
                            nn.ReLU(inplace=True),
                        )
                    )
                    inch = outc
                layers.append(nn.Sequential(*convs))
        return nn.ModuleList(layers)

    def _make_stage(self, stage_cfg, num_inchannels, multi_scale_output=True):
        modules = []
        block = blocks_dict[stage_cfg.block]
        num_channels = [c * block.expansion for c in stage_cfg.num_channels]
        for i in range(stage_cfg.num_modules):
            reset_multi_scale_output = (
                multi_scale_output or i != stage_cfg.num_modules - 1
            )
            modules.append(
                HRModule(
                    stage_cfg.num_branches,
                    block,
                    stage_cfg.num_blocks,
                    num_inchannels,
                    num_channels,
                    stage_cfg.fuse_method,
                    reset_multi_scale_output,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        x_list = [t(x) if t else x for t in self.transition1]
        y_list = self.stage2(x_list)

        x_list = [
            t(y_list[-1]) if t else y_list[i] for i, t in enumerate(self.transition2)
        ]
        y_list = self.stage3(x_list)

        x_list = [
            t(y_list[-1]) if t else y_list[i] for i, t in enumerate(self.transition3)
        ]
        y_list = self.stage4(x_list)

        x = self.final_layer(y_list[0])
        return F.sigmoid(x)


def compute_flops(model: nn.Module, input_tensor: torch.Tensor):
    device = next(model.parameters()).device
    x = input_tensor.to(device)

    total_flops = 0

    def conv_hook(module, input, output):
        B, C_out, H_out, W_out = output.shape
        C_in = module.in_channels
        K_h, K_w = module.kernel_size
        groups = module.groups
        flops = 2 * B * C_out * H_out * W_out * (C_in // groups) * K_h * K_w
        nonlocal total_flops
        total_flops += flops

    def linear_hook(module, input, output):
        B, in_features = input[0].shape
        out_features = module.out_features
        flops = 2 * B * in_features * out_features
        nonlocal total_flops
        total_flops += flops

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    return round(total_flops / 1e9, 2)


def print_model(model: HRNet, input_size: tuple[int] = (1, 3, 256, 192)):
    from torchinfo import summary

    device = next(model.parameters()).device
    input_tensor = torch.randn(input_size).to(device)
    summary(
        model,
        input_size=input_size,
        col_names=("input_size", "output_size", "num_params", "trainable"),
        depth=2,
        verbose=1,
        device=device,
    )

    print(f"HRNet summary:", end=" ")
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters", end=", ")
    print(f"{compute_flops(model, input_tensor)} GFLOPs")


if __name__ == "__main__":
    from config import HRNetConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = HRNetConfig()
    model = HRNet(config).to(device)
    print_model(model, (1, 3, config.model.input_height, config.model.input_width))
    input_tensor = torch.randn(
        (1, 3, config.model.input_height, config.model.input_width)
    ).to(device)
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output min: {output.min().item()}, max: {output.max().item()}")
