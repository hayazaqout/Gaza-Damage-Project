"""Compact six-channel U-Net (paper Sec. 3.3).

Early-fusion architecture: before-RGB and after-RGB are concatenated into a
single 6-channel input from the first convolutional layer, rather than being
compared via a Siamese/feature-level fusion (paper's Appendix A describes
evaluating and abandoning a Siamese formulation during development).

Reconstructed from the paper's architecture description (encoder ~24/48/96/192
filters, 256-filter bottleneck, symmetric decoder with skip connections, batch
norm, ReLU, spatial dropout in deeper stages, 128x128x6 input). The paper
reports 3,315,345 parameters; this reconstruction lands at 3,311,441 (-0.1%),
verified locally with a CPU-only forward pass (see repo notes) -- close enough
to trust as a faithful reconstruction of the described architecture, though it
is necessarily a reconstruction rather than the authors' exact original code.
"""
import torch
import torch.nn as nn

from . import config


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SixChannelUNet(nn.Module):
    def __init__(self, in_ch=config.MODEL_IN_CHANNELS,
                 filters=config.MODEL_ENCODER_FILTERS,
                 bottleneck=config.MODEL_BOTTLENECK_FILTERS,
                 dropout=config.MODEL_DROPOUT):
        super().__init__()
        f0, f1, f2, f3 = filters
        self.enc0 = DoubleConv(in_ch, f0)
        self.enc1 = DoubleConv(f0, f1)
        self.enc2 = DoubleConv(f1, f2)
        self.enc3 = DoubleConv(f2, f3, dropout=dropout)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(f3, bottleneck, dropout=dropout)

        self.up3 = nn.ConvTranspose2d(bottleneck, f3, 2, stride=2)
        self.dec3 = DoubleConv(f3 + f3, f3, dropout=dropout)
        self.up2 = nn.ConvTranspose2d(f3, f2, 2, stride=2)
        self.dec2 = DoubleConv(f2 + f2, f2)
        self.up1 = nn.ConvTranspose2d(f2, f1, 2, stride=2)
        self.dec1 = DoubleConv(f1 + f1, f1)
        self.up0 = nn.ConvTranspose2d(f1, f0, 2, stride=2)
        self.dec0 = DoubleConv(f0 + f0, f0)

        self.out_conv = nn.Conv2d(f0, 1, kernel_size=1)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), e0], dim=1))
        return self.out_conv(d0)  # raw logits, (N, 1, H, W)
