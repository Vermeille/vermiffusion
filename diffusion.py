import torch


def b(x, dim):
    return x.view([-1] + [1] * (dim - max(1, x.dim())))


class ForwardProcess(torch.nn.Module):

    def __init__(self, beta):
        super().__init__()
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', 1 - beta)
        self.register_buffer('alpha_bar', torch.cumprod(self.alpha, dim=0))

    def forward_to(self, x0, e, t):
        a_bar_t = b(self.alpha_bar[t], 4)
        return (torch.sqrt(a_bar_t) * x0 + torch.sqrt(1 - a_bar_t) * e)

    def snr(self):
        return self.alpha_bar.log() - (1 - self.alpha_bar).log()

    def v_target(self, x0, e, t):
        a_bar_t = b(self.alpha_bar[t], 4)
        return (torch.sqrt(a_bar_t) * e - torch.sqrt(1 - a_bar_t) * x0)

    def make_targets(self, x0, t):
        e = torch.randn_like(x0)
        return {
            'xt': self.forward_to(x0, e, t),
            'v': self.v_target(x0, e, t),
            'e': e
        }

    def to_x0(self, xt, v, t):
        a_bar_t = b(self.alpha_bar[t], 4)
        return torch.sqrt(a_bar_t) * xt - torch.sqrt(1 - a_bar_t) * v

    def to_noise(self, xt, v, t):
        a_bar_t = b(self.alpha_bar[t], 4)
        return torch.sqrt(1 - a_bar_t) * xt + torch.sqrt(a_bar_t) * v


def get_cosine(T, off=0.008, pow=3):
    f = torch.cos(torch.linspace(off, 1 + off, T) / (1 + off) * torch.pi /
                  2)**pow
    f = f / f[0]
    beta = 1 - f[1:] / f[:-1]
    beta = torch.cat([torch.tensor([0]), beta], 0)
    return ForwardProcess(beta)


if __name__ == '__main__':
    cos = get_cosine(100)
    print(cos.snr())
    print('hello world')
    from PIL import Image
    import torchvision.transforms as TF
    im = Image.open(
        '~/pinterest-downloader/faces/Arkhadtoa/Monster concept art/501940320971110709_Mega Dark Things.png-0-female.jpg'
    )
    im = TF.functional.to_tensor(im)[None]
    cos.forward_to(im, torch.randn_like(im), 0)
