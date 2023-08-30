from collections import OrderedDict
import torchelie as tch
import torchelie.utils as tu
import torchelie.nn as tnn
import torch
import torch.nn as nn
import torch.nn.functional as F


class Scale(nn.Module):

    def __init__(self, scale):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, x):
        return x * self.scale


class ResidualSA(nn.Module):

    def __init__(self, ch):
        super().__init__()
        self.branch = nn.Sequential(
            OrderedDict([('norm', nn.GroupNorm(1, ch, affine=True)),
                         ('sa',
                          tnn.SelfAttention2d(ch, ch // 64, checkpoint=False)),
                         ('scale', Scale(0.01))]))

    def forward(self, x):
        return x + self.branch(x)


class ResBlock(tnn.CondSeq):

    def __init__(self, in_channels, out_channels):
        super().__init__(
            OrderedDict([
                ('norm', nn.GroupNorm(1, in_channels, affine=False)),
                ('temb', tnn.FiLM2d(in_channels, 1024)),
                ('act', nn.GELU()),
                ('conv',
                 tu.kaiming(tnn.Conv3x3(in_channels, out_channels,
                                        bias=False))),
                ('norm2', nn.GroupNorm(1, out_channels, affine=False)),
                ('temb2', tnn.FiLM2d(out_channels, 1024)),
                ('act2', nn.GELU()),
                ('conv2',
                 tu.kaiming(tnn.Conv3x3(out_channels, out_channels,
                                        bias=False))),
                ('scale', Scale(0.01)),
            ]))

    def forward(self, x, y=None):
        return tnn.CondSeq.forward(self, x, y) + x


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        base_ch = 32
        self.temb = nn.Sequential(
            tu.normal_init(nn.Embedding(1000, 1024), std=0.002),
            nn.GroupNorm(1, 1024))
        downs = [tnn.CondSeq(tnn.Conv1x1(in_channels, base_ch), nn.GELU())]
        ups = [
            tnn.CondSeq(
                nn.GroupNorm(1, base_ch),
                tu.normal_init(tnn.Conv1x1(base_ch, out_channels), 0.02))
        ]

        factor = 2
        base_ch = int(base_ch * factor)
        for i in range(4):
            ch = int(factor**i * base_ch)
            print(ch)
            downs.append(
                tnn.CondSeq(
                    OrderedDict([
                        ('norm',
                         nn.GroupNorm(1, int(ch // factor), affine=False)),
                        ('conv',
                         tu.kaiming(
                             nn.Conv2d(int(ch // factor),
                                       ch,
                                       2,
                                       stride=2,
                                       bias=False))),
                        ('sa1', ResidualSA(ch) if i > 1 else nn.Identity()),
                        ('res1', ResBlock(ch, ch)),
                        ('sa2', ResidualSA(ch) if i > 1 else nn.Identity()),
                        ('res2', ResBlock(ch, ch)),
                        ('res3', ResBlock(ch, ch)),
                    ])))

            ups.append(
                tnn.CondSeq(
                    OrderedDict([
                        ('sa1', ResidualSA(ch) if i > 1 else nn.Identity()),
                        ('res1', ResBlock(ch, ch)),
                        ('sa2', ResidualSA(ch) if i > 1 else nn.Identity()),
                        ('res2', ResBlock(ch, ch)),
                        ('res3', ResBlock(ch, ch)),
                        ('up', nn.Upsample(scale_factor=2)),
                        ('conv',
                         tu.kaiming(
                             tnn.Conv3x3(ch, int(ch // factor), bias=False))),
                        ('scale', Scale(0.0)),
                    ])))
        self.downs = nn.ModuleList(downs)
        self.ups = nn.ModuleList(ups)

    def forward(self, x, y):
        y = self.temb(y)
        skips = []
        for down in self.downs:
            x = down(x, y)
            skips.append(x)
        skips[-1] = torch.zeros_like(skips[-1])

        for up, skip in zip(self.ups[::-1], skips[::-1]):
            x = x + skip
            x = up(x, y)

        return x


def get_model():
    return tch.models.UNet([3, 32, 64, 128, 256, 512], 3)


if __name__ == '__main__':
    #m = get_model()
    m = UNet(3, 3)
    m.downs[0](torch.randn(1, 3, 256, 256), torch.randint(0, 1000, (1, )))
    print(m)
    print(m(torch.randn(1, 3, 256, 256), torch.randint(0, 1000, (1, ))).shape)
