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
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class ResBlock(tnn.CondSeq):

    def __init__(self, in_channels, out_channels):
        super().__init__(
            OrderedDict([
                ('norm', nn.GroupNorm(32, in_channels, affine=False)),
                ('temb', tnn.FiLM2d(in_channels, 1024)),
                ('act', nn.ReLU()),
                ('conv', tnn.Conv3x3(in_channels, out_channels)),
                ('norm2', nn.GroupNorm(32, out_channels, affine=False)),
                ('temb2', tnn.FiLM2d(out_channels, 1024)),
                ('act2', nn.ReLU()),
                ('conv2', tnn.Conv3x3(out_channels, out_channels)),
                ('scale', Scale(0.1)),
            ]))

    def forward(self, x, y=None):
        return tnn.CondSeq.forward(self, x, y) + x


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        base_ch = 32
        self.temb = tu.normal_init(nn.Embedding(1000, 1024))
        downs = [tnn.CondSeq(tnn.Conv1x1(in_channels, base_ch))]
        ups = [
            tnn.CondSeq(nn.GroupNorm(1, base_ch),
                        tnn.Conv1x1(base_ch, out_channels))
        ]

        base_ch = base_ch * 2
        for i in range(4):
            ch = 2**i * base_ch
            print(ch)
            downs.append(
                tnn.CondSeq(
                    OrderedDict([
                        ('conv', tnn.Conv3x3(ch // 2, ch, stride=2,
                                             bias=False)),
                        ('res1', ResBlock(ch, ch)),
                        ('res2', ResBlock(ch, ch)),
                    ])))

            ups.append(
                tnn.CondSeq(
                    OrderedDict([
                        ('res1', ResBlock(ch, ch)),
                        ('res2', ResBlock(ch, ch)),
                        ('up', nn.Upsample(scale_factor=2)),
                        ('conv', tnn.Conv3x3(ch, ch // 2, bias=False)),
                        ('scale', Scale(0.1)),
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
