# LSKA + SPPF-LSKA：大核可分离注意力
import torch
import torch.nn as nn
from .conv import Conv


class LSKA(nn.Module):
    """Large Separable Kernel Attention (LSKNet)。

    用水平+垂直的深度可分离卷积 + 膨胀卷积，以低成本获得大感受野，
    生成空间注意力图增强特征。最后一层零初始化，初始退化为恒等映射，
    便于从 P2 权重热启动（初始等价于原 SPPF）。
    """

    def __init__(self, dim, k_size=7):
        super().__init__()
        self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, k_size), padding=(0, k_size // 2), groups=dim)
        self.conv0v = nn.Conv2d(dim, dim, kernel_size=(k_size, 1), padding=(k_size // 2, 0), groups=dim)
        self.conv_sh = nn.Conv2d(dim, dim, kernel_size=(1, k_size), padding=(0, k_size - 1), groups=dim, dilation=(1, 2))
        self.conv_sv = nn.Conv2d(dim, dim, kernel_size=(k_size, 1), padding=(k_size - 1, 0), groups=dim, dilation=(2, 1))
        self.conv1 = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)

    def forward(self, x):
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_sh(attn)
        attn = self.conv_sv(attn)
        attn = self.conv1(attn)
        return x * (2.0 * torch.sigmoid(attn))  # 初始=恒等，权重有界 (0,2)


class SPPF_LSKA(nn.Module):
    """SPPF-LSKA：SPPF 后接大核可分离注意力。"""

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.lska = LSKA(c_ * 4)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        x = torch.cat(y, 1)
        x = self.lska(x)
        return self.cv2(x)
