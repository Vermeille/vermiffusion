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
                          tnn.SelfAttention2d(ch, ch // 32, checkpoint=False)),
                         ('scale', Scale(0.01))]))

    def forward(self, x):
        return x + self.branch(x)


class ResBlock(tnn.CondSeq):

    def __init__(self, in_channels, out_channels):
        super().__init__(
            OrderedDict([
                ('norm', nn.GroupNorm(1, in_channels, affine=True)),
                ('temb', tnn.FiLM2d(in_channels, 1024)),
                #('act', nn.GELU()),
                ('conv',
                 tu.kaiming(tnn.Conv3x3(in_channels, out_channels,
                                        bias=False))),
                #('norm2', nn.GroupNorm(1, out_channels, affine=False)),
                #('temb2', tnn.FiLM2d(out_channels, 1024)),
                ('act2', nn.GELU()),
                ('conv2',
                 tu.kaiming(tnn.Conv3x3(out_channels, out_channels,
                                        bias=False))),
                ('scale', Scale(0.01)),
            ]))

    def forward(self, x, y=None):
        return tnn.CondSeq.forward(self, x, y) + x


class FourierFeatures2d(nn.Module):
    def __init__(self, n_features, min, max):
        super().__init__()
        self.n_features = n_features
        self.min = min
        self.max = max

    def forward(self, x):
        N, C, H, W = x.shape
        x_ff = x.unsqueeze(1)  # N, F, C, H, W
        scales = 2**(torch.linspace(self.min, self.max, self.n_features, device=x.device))
        scales = scales.view(-1, 1, 1, 1)
        x_ff = x_ff * scales * torch.pi
        x_ff = x_ff.view(N, -1, H, W)
        return torch.cat((x, torch.cos(x_ff), torch.sin(x_ff)), dim=1)



class UNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        chs = [128, 128, 256, 512, 512]
        blocks = [2, 2, 4, 4, 4]
        self.temb = nn.Sequential(
            tu.normal_init(nn.Embedding(1000, 1024), std=0.02),
            nn.GroupNorm(1, 1024)
            )
        downs = [
            tnn.CondSeq(
                FourierFeatures2d(8, 1, 8),
                tnn.Conv1x1(3 + 3 * 8 * 2, chs[0])
            #, nn.GELU()
            )]
        ups = [
            tnn.CondSeq(
                nn.GroupNorm(1, chs[0]),
                tu.normal_init(tnn.Conv1x1(chs[0], out_channels), 0.02),
                Scale(1.0))
        ]

        
        for i in range(len(chs)):
            ch = chs[i]
            print(ch)
            min_sa = 1
            downs.append(
                tnn.CondSeq(
                    OrderedDict([
                        ('norm', nn.GroupNorm(1, chs[max(0, i - 1)], affine=False)),
                        ('conv', tu.kaiming( nn.Conv2d(chs[max(0, i - 1)], ch, 2, stride=2, bias=False))),
                        #('down', nn.AvgPool2d(3, 2, 1)),
                        ('sa1', ResidualSA(ch) if i > min_sa else nn.Identity()),
                        ('res1', ResBlock(ch, ch)),
                        ('sa2', ResidualSA(ch) if i > min_sa else nn.Identity()),
                        ('res2', ResBlock(ch, ch)),
                        ('sa3', ResidualSA(ch) if i > min_sa else nn.Identity()),
                        ('res3', ResBlock(ch, ch)),
                    ])))

            ups.append(
                tnn.CondSeq(
                    OrderedDict([
                        ('norm', nn.GroupNorm(1, int(ch), affine=False)),
                        ('sa1', ResidualSA(ch) if i > min_sa else nn.Identity()),
                        ('res1', ResBlock(ch, ch)),
                        ('sa2', ResidualSA(ch) if i > min_sa else nn.Identity()),
                        ('res2', ResBlock(ch, ch)),
                        ('sa3', ResidualSA(ch) if i > min_sa else nn.Identity()),
                        ('res3', ResBlock(ch, ch)),
                        ('up', nn.Upsample(scale_factor=2)),
                        ('conv', tu.kaiming( tnn.Conv3x3(ch, chs[max(0, i - 1)], bias=False))),
                        #('scale', Scale(0.1)),
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
