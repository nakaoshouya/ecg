def N(ch): return nn.GroupNorm(num_groups=8, num_channels=ch)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            N(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            N(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, num_classes=2, base_ch=16):
        super().__init__()
        self.inc   = DoubleConv(in_ch, base_ch)          # 1 -> 64
        self.down1 = Down(base_ch, base_ch*2)            # 64 -> 128
        self.down2 = Down(base_ch*2, base_ch*4)          # 128 -> 256
        self.down3 = Down(base_ch*4, base_ch*8)          # 256 -> 512
        self.down4 = Down(base_ch*8, base_ch*16)         # 512 -> 1024 (bottleneck)

        self.up1 = Up(base_ch*16, base_ch*8)             # 1024 -> 512
        self.up2 = Up(base_ch*8,  base_ch*4)             # 512  -> 256
        self.up3 = Up(base_ch*4,  base_ch*2)             # 256  -> 128
        self.up4 = Up(base_ch*2,  base_ch)               # 128  -> 64

        self.outc = nn.Conv2d(base_ch, num_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits