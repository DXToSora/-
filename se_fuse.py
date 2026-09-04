# SEFuse: SE 通道注意力增强的特征融合（Concat 的严格超集）
# 用法: 在 yaml 中把 PANet 的 Concat 换成 SEFuse
import torch
import torch.nn as nn


class SEFuse(nn.Module):
    """SE 通道注意力增强的特征融合。

    对两路输入分别做通道注意力（Squeeze-and-Excitation）加权后，按通道拼接。
    注意力支路最后一层零初始化，使初始权重为 1，因此初始状态完全等价于普通
    Concat；训练中可学到非平凡权重以超越 Concat（容量上是 Concat 的严格超集）。

    Args:
        c1 (int): 第一路输入通道数
        c2 (int): 第二路输入通道数
        r (int): SE 压缩比
    """

    def __init__(self, c1, c2, r=8):
        super().__init__()
        self.se1 = self._make_se(c1, r)
        self.se2 = self._make_se(c2, r)

    def _make_se(self, c, r):
        hidden = max(c // r, 4)
        se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, c, 1),
        )
        # 零初始化最后一层 -> 初始输出 0 -> 权重 1+0=1 -> 等价 Concat
        nn.init.zeros_(se[-1].weight)
        nn.init.zeros_(se[-1].bias)
        return se

    def forward(self, x):
        a, b = x
        return torch.cat([a * (1.0 + self.se1(a)), b * (1.0 + self.se2(b))], 1)
