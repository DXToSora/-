# HBS: Hierarchical Background Smoothing (CVPR 2025, SET)
# 用 GT 框生成二值 mask 把 FPN 特征解耦成前景/背景，对背景做
# "降维→ReLU→升维 + 残差"的自适应平滑以抑制高频噪声，再重组。
# 论文 Eq.2: P_E = P_fg + φ(P_bg, r) = P⊛M + φ(P⊛¬M, r)
# 论文 Eq.3: φ(P_bg, r) = σ(w_e ⊗ σ(w_r ⊗ P_bg)) + P_bg
#
# 注意：HBS 依赖 GT mask，推理时无 GT，因此仅作为【训练时的辅助监督】使用，
# 推理路径不调用本模块、零开销。
import math

import torch
import torch.nn as nn

from ultralytics.utils.ops import xywh2xyxy


def kernel_by_stride(s):
    """核大小按论文 Eq.4: K = 2*floor(log2(s)/2)+1，保证奇数。

    P2(s=4)→3, P3(s=8)→3, P4(s=16)→5, P5(s=32)→5
    """
    return int(math.floor(math.log2(s) / 2) * 2 + 1)


class HBS(nn.Module):
    """单层背景平滑模块（论文 Eq.3）。

    Args:
        c (int): 输入/输出通道数。
        r (int): 通道压缩比（论文 r=4）。
        k (int): 卷积核大小（由 stride 决定，见 kernel_by_stride）。
    """

    def __init__(self, c, r=4, k=3):
        super().__init__()
        hidden = max(c // r, 1)
        self.conv1 = nn.Conv2d(c, hidden, k, padding=k // 2)
        self.act = nn.ReLU(inplace=True)  # 论文 σ 为 ReLU
        self.conv2 = nn.Conv2d(hidden, c, k, padding=k // 2)
        # 零初始化升维层 -> 初始 φ(P_bg)=0 -> 增强特征退化为原始特征（热启动，训练更稳）
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, p, mask):
        """p: (B,C,H,W)，mask: (B,1,H,W) 0/1。"""
        p_fg = p * mask
        p_bg = p * (1.0 - mask)
        p_bg_s = self.conv2(self.act(self.conv1(p_bg))) + p_bg
        return p_fg + p_bg_s


class HBSAux(nn.Module):
    """HBS + 独立辅助检测头（仅训练时调用）。

    Args:
        ch_list (list[int]): 各检测层(P2~P5)的通道数。
        strides (list[int]): 各检测层的 stride（决定核大小）。
        nc (int): 类别数。
        r (int): 通道压缩比。
    """

    def __init__(self, ch_list, strides, nc, r=4):
        super().__init__()
        from .head import Detect  # 延迟导入，避免循环依赖

        self.hbs = nn.ModuleList(HBS(c, r, kernel_by_stride(s)) for c, s in zip(ch_list, strides))
        # 独立辅助头：训练时只用 Detect.forward 的 cv2/cv3 拼接路径，不依赖 stride/anchor。
        # DFL 无参数(requires_grad_(False))，不引入额外可训练参数。
        self.aux_head = Detect(nc=nc, ch=tuple(ch_list))

    def forward(self, neck_feats, masks):
        enhanced = [self.hbs[i](neck_feats[i], masks[i]) for i in range(len(neck_feats))]
        return self.aux_head(enhanced)


def build_gt_mask(batch, neck_feats):
    """由 GT 框生成每层硬二值 mask（框内=1，框外=0）。

    GT 框为归一化 xywh（batch["bboxes"], [N,4]），乘以特征图宽高对齐到各层分辨率。
    坐标取整：左上 floor、右下 ceil，保证 tiny box 至少覆盖 1px、不丢前景。

    Args:
        batch (dict): 含 bboxes([N,4] 归一化 xywh)、batch_idx([N,1])。
        neck_feats (list[Tensor]): 各检测层特征 (B,C,H,W)。

    Returns:
        list[Tensor]: 每层 (B,1,H,W) 的 0/1 mask。
    """
    device = neck_feats[0].device
    bboxes = batch["bboxes"].to(device)
    # 注意：dataloader 的 batch_idx 默认是 float32（torch.zeros 无 dtype），必须转 long
    batch_idx = batch["batch_idx"].to(device).view(-1).long()
    batch_size = neck_feats[0].shape[0]

    masks = []
    for feat in neck_feats:
        _, _, h, w = feat.shape
        mask = torch.zeros(batch_size, 1, h, w, device=device)
        if bboxes.numel():
            xyxy = xywh2xyxy(bboxes)  # (N,4) 归一化 xyxy
            xyxy = xyxy * torch.tensor([w, h, w, h], device=device, dtype=xyxy.dtype)
            x1 = xyxy[:, 0].floor().clamp(0, w).long()
            y1 = xyxy[:, 1].floor().clamp(0, h).long()
            x2 = xyxy[:, 2].ceil().clamp(0, w).long()
            y2 = xyxy[:, 3].ceil().clamp(0, h).long()
            for n in range(bboxes.shape[0]):
                b = int(batch_idx[n])
                x1n, y1n = int(x1[n]), int(y1[n])
                x2n, y2n = int(x2[n]), int(y2[n])
                if x2n > x1n and y2n > y1n:  # 跳过空框（下采样后尺寸<=0）
                    mask[b, 0, y1n:y2n, x1n:x2n] = 1.0
        masks.append(mask)
    return masks
