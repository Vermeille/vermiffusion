import torchelie as tch
import torchelie.utils as tu
import torchelie.nn as tnn
import torchelie.callbacks as tcb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as TF

from sampler import DDIM
from diffusion import get_cosine
from model import UNet


def main(rank, world_size):
    basem = UNet(3, 3).to(rank)
    #basem.load_state_dict(torch.load('model/ckpt_32000.pth', map_location='cpu')['model'])
    #basem = torch.compile(basem)
    m = torch.nn.parallel.DistributedDataParallel(basem,
                                                  device_ids=[rank],
                                                  output_device=rank)

    diff = get_cosine(1000).to(rank)

    print(next(iter(m.parameters())).device)

    dat = tch.datasets.UnlabeledImages('~/pinterest-downloader/faces/',
                                       transform=TF.Compose([
                                           TF.Resize(96),
                                           TF.CenterCrop(96),
                                           TF.RandomHorizontalFlip(),
                                           TF.ToTensor(),
                                           TF.Normalize([0.5] * 3, [0.5] * 3)
                                       ]))

    data_loader = torch.utils.data.DataLoader(dat,
                                              batch_size=48,
                                              shuffle=True,
                                              num_workers=4,
                                              pin_memory=True)

    def train(x):
        x = x[0]
        x0 = x.to(rank)
        t = torch.randint(0, 1000, (x.size(0), ), device=rank)
        tgt = diff.make_targets(x0, t)
        #with torch.autocast('cuda', dtype=torch.float16):
        pred = m(tgt['xt'], t).float()
        loss = F.mse_loss(pred, tgt['v'])
        loss.backward()
        return {
            'loss': loss,
            'x0': diff.to_x0(tgt['xt'], pred, t).clamp(-1, 1),
            'xt': tgt['xt'].clamp(-1, 1)
        }

    def test():
        g = torch.Generator(device=rank)
        g.manual_seed(327023487 + rank)
        ddim = DDIM(diff)
        with torch.autocast('cuda'):
            xt = ddim.sample(basem, (16, 3, 96, 96), 21, eta=0,
                             generator=g).float()
        return {'gen': xt.clamp(-1, 1)}

    LR = 1e-3
    opt = torch.optim.AdamW(m.parameters(),
                            lr=LR,
                            betas=(0.9, 0.95),
                            weight_decay=0.01)

    recipe = tch.recipes.TrainAndCall(
        m,
        train,
        test,
        data_loader,
        test_every=500,
        visdom_env=f'vermiffusion_{LR}' if rank == 0 else None)

    recipe.callbacks.add_callbacks([
        tcb.Optimizer(opt),
        tcb.Log('loss', 'loss'),
        tcb.Log('x0', 'pred_x0'),
        tcb.Log('xt', 'xt'),
        tcb.Log('batch.0', 'batch'),
    ])
    recipe.test_loop.callbacks.add_callbacks([
        tcb.Log('gen', 'gen'),
    ])
    recipe.to(rank)
    recipe.run(10)


if __name__ == '__main__':
    import argparse
    tu.parallel_run(main)
