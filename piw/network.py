"""The Person-in-WiFi network (paper Figure 6), pose heads only.

Data flow (per CLAUDE.md):
  input  150 x 3 x 3          5 time samples x 30 subcarriers = 150 channels,
                              3 x 3 = transmit x receive antenna pairs
  bilinear upsample -> 150 x 96 x 96
  residual conv block (keeps 150 x 96 x 96)
  U-Net, one shared trunk -> base x 96 x 96
  two heads, each downsampled to c x 46 x 82 with a strided conv:
     JHM  19 x 46 x 82   (18 joints + background)
     PAF  38 x 46 x 82   (19 limbs x 2)

The paper's third head (segmentation) is dropped: Wi-Pose has no masks.

Head kernel derivation. To reach 46 x 82 from 96 x 96 with "stride 2 on
height, stride 1 on width" (CLAUDE.md) using one valid conv:
    height: (96 - kh) / 2 + 1 = 46  ->  kh = 6
    width:  (96 - kw) / 1 + 1 = 82  ->  kw = 15
so kernel (6, 15), stride (2, 1), no padding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from piw.skeleton import JHM_CHANNELS, PAF_CHANNELS

IN_CHANNELS = 150     # 5 time samples x 30 subcarriers
UPSAMPLE_HW = 96      # paper upsamples the 3x3 antenna grid to 96x96
OUT_H, OUT_W = 46, 82


class DoubleConv(nn.Module):
    """(3x3 conv -> BN -> ReLU) x 2, spatial size unchanged."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    """Two 3x3 convs with an identity skip; channels and size unchanged."""

    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.body(x))


class UNet(nn.Module):
    """Small symmetric U-Net; input and output share spatial size.

    Depth-3 encoder/decoder with skip connections. At 96x96 the pooling path
    is 96 -> 48 -> 24 -> 12 and back, all exact integer halvings.
    """

    def __init__(self, in_ch, base=64):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.bottleneck = DoubleConv(base * 4, base * 8)
        self.dec3 = DoubleConv(base * 8 + base * 4, base * 4)
        self.dec2 = DoubleConv(base * 4 + base * 2, base * 2)
        self.dec1 = DoubleConv(base * 2 + base, base)
        self.pool = nn.MaxPool2d(2)
        self.out_channels = base

    @staticmethod
    def _up(x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                          align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(self._up(b, e3))
        d2 = self.dec2(self._up(d3, e2))
        d1 = self.dec1(self._up(d2, e1))
        return d1


class Head(nn.Module):
    """Refine at 96x96, then a strided conv down to out_ch x 46 x 82."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
        )
        self.down = nn.Conv2d(in_ch, out_ch, kernel_size=(6, 15),
                              stride=(2, 1))

    def forward(self, x):
        return self.down(self.refine(x))


class PersonInWiFi(nn.Module):
    """Full pose network: CSI tensor -> (JHM, PAF).

    Parameters
    ----------
    base : channel width of the U-Net's first stage.
    jhm_channels, paf_channels : head output channels; default to the
        Wi-Pose COCO-18 counts (19 and 38) taken from piw.skeleton.
    """

    def __init__(self, base=64, jhm_channels=JHM_CHANNELS,
                 paf_channels=PAF_CHANNELS):
        super().__init__()
        self.res_block = ResidualBlock(IN_CHANNELS)
        self.unet = UNet(IN_CHANNELS, base=base)
        self.jhm_head = Head(self.unet.out_channels, jhm_channels)
        self.paf_head = Head(self.unet.out_channels, paf_channels)

    @staticmethod
    def upsample_input(x):
        """Bilinear upsample the 3x3 antenna grid to 96x96 (150 channels)."""
        return F.interpolate(x, size=(UPSAMPLE_HW, UPSAMPLE_HW),
                             mode="bilinear", align_corners=False)

    def forward(self, x):
        x = self.upsample_input(x)      # (B, 150, 96, 96)
        x = self.res_block(x)           # (B, 150, 96, 96)
        trunk = self.unet(x)            # (B, base, 96, 96)
        jhm = self.jhm_head(trunk)      # (B, 19, 46, 82)
        paf = self.paf_head(trunk)      # (B, 38, 46, 82)
        return jhm, paf


def count_parameters(model):
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
