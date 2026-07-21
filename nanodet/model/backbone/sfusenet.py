import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .shufflenetv2 import ShuffleNetV2


class SequentialFusionAttention(nn.Module):
    def __init__(self, in_channels, groups=4, reduction_ratio=8, kernel_size=7):
        super().__init__()
        self.groups = max(1, math.gcd(in_channels, groups))
        self.group_channels = in_channels // self.groups
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

        self.group_modules = nn.ModuleList(
            [
                self._make_group_attention(
                    self.group_channels, reduction_ratio, kernel_size
                )
                for _ in range(self.groups)
            ]
        )

    def _make_group_attention(self, in_ch, reduction_ratio, kernel_size):
        reduced_ch = max(in_ch // reduction_ratio, 4)
        channel_mlp = nn.Sequential(
            nn.Linear(in_ch, reduced_ch, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_ch, in_ch, bias=False),
            nn.Sigmoid(),
        )
        spatial_conv_h = nn.Conv2d(
            1, 1, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0), bias=False
        )
        spatial_conv_w = nn.Conv2d(
            1, 1, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2), bias=False
        )
        return nn.ModuleDict(
            {
                "channel_mlp": channel_mlp,
                "spatial_conv_h": spatial_conv_h,
                "spatial_conv_w": spatial_conv_w,
            }
        )

    def forward(self, x):
        chunks = torch.chunk(x, self.groups, dim=1)
        outputs = []

        for xi, att in zip(chunks, self.group_modules):
            b, c, _, _ = xi.size()
            avg_pool = F.adaptive_avg_pool2d(xi, 1).view(b, c)
            ca_weight = att["channel_mlp"](avg_pool).view(b, c, 1, 1)
            xi_ca = xi * ca_weight

            avg_map = torch.mean(xi_ca, dim=1, keepdim=True)
            sa_h = att["spatial_conv_h"](avg_map)
            sa_w = att["spatial_conv_w"](avg_map)
            sa_weight = torch.sigmoid(sa_h + sa_w)

            outputs.append(xi_ca * sa_weight)

        return torch.cat(outputs, dim=1)


class SFuseNet(nn.Module):
    """
    NanoDet-compatible backbone.

    This wraps NanoDet's ShuffleNetV2 and refines each output stage with
    Sequential Fusion Attention while preserving the tuple-of-features API that
    the FPN expects.
    """

    def __init__(
        self,
        model_size="1.0x",
        out_stages=(2, 3, 4),
        activation="LeakyReLU",
        pretrain=True,
        groups=4,
        reduction_ratio=8,
    ):
        super().__init__()
        self.backbone = ShuffleNetV2(
            model_size=model_size,
            out_stages=out_stages,
            activation=activation,
            pretrain=pretrain,
        )
        stage_channels = {
            "0.5x": [48, 96, 192],
            "1.0x": [116, 232, 464],
            "1.5x": [176, 352, 704],
            "2.0x": [244, 488, 976],
        }[model_size]
        self.sfa = nn.ModuleList(
            [
                SequentialFusionAttention(
                    in_channels=channels,
                    groups=groups,
                    reduction_ratio=reduction_ratio,
                )
                for channels in stage_channels
            ]
        )

    def forward(self, x):
        feats = self.backbone(x)
        return tuple(block(feat) for block, feat in zip(self.sfa, feats))
