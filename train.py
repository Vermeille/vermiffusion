import torchelie as tch
import torchelie.utils as tu
import torchelie.nn as tnn
import torchelie.callbacks as tcb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as TF

from sampler import DDIM, FlowSampler
from diffusion import get_cosine, get_linear, FlowMatching
from model import UNet
import schedulefree


def main(tag, rank, world_size):
    basem = UNet(3, 3).to(rank)
    #basem = torch.compile(basem)
    #basem.load_state_dict(torch.load('model/ckpt_3000.pth', map_location='cpu')['model'])
    if rank == 0:
        print(basem)
    if world_size > 1:
        m = torch.nn.parallel.DistributedDataParallel(basem,
                                                  device_ids=[rank],
                                                  output_device=rank)
    else:
        m = basem

    alg = 'flow'
    if alg == 'diffusion':
        diff = get_cosine(1000).to(rank)
        sampler = DDIM(diff)
    elif alg == 'flow':
        diff = FlowMatching(1000)
        sampler = FlowSampler(diff)
    else:
        assert False

    dat = tch.datasets.UnlabeledImages('~/faces/',
                                       transform=TF.Compose([
                                           TF.Resize(96),
                                           TF.CenterCrop(96),
                                           TF.RandomHorizontalFlip(),
                                           TF.ToTensor(),
                                           TF.Normalize([0.5] * 3, [0.5] * 3)
                                       ]))

    #dat = [dat[i % 32] for i in range(512-128)]
    #dat = [dat[0] for i in range(512-128)]
    data_loader = torch.utils.data.DataLoader(dat,
                                              batch_size=512 - 128 - 32,
                                              shuffle=True,
                                              num_workers=4,
                                              pin_memory=True,
                                              persistent_workers=True)

    def train(x):
        x = x[0]
        x0 = x.to(rank)
        t = torch.randint(0, diff.T, (x.size(0), 2), device=rank).max(dim=1).values
        t = t.sort().values
        tgt = diff.make_targets(x0, t)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = m(tgt['xt'], t).float()
        loss = F.l1_loss(diff.to_x0(tgt['xt'], pred, t), x0)
        #loss = F.mse_loss(pred, tgt['v'])
        loss.backward()
        return {
            'loss': loss,
            'x0': diff.to_x0(tgt['xt'], pred, t).clamp(-1, 1),
            'xt': tgt['xt'].clamp(-1, 1)
        }

    def test():
        opt.eval()
        g = torch.Generator(device=rank)
        g.manual_seed(327023487 + rank)
        xt = sampler.sample(basem, (24, 3, 96, 96), 51, eta=0,
                         generator=g).float()
        opt.train()
        return {'gen': xt.clamp(-1, 1)}

    LR = 2e-3
    EPOCHS = 20
    #opt = torch.optim.AdamW(m.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=0.001)
    opt = schedulefree.AdamWScheduleFree(m.parameters(), LR, warmup_steps=50, weight_decay=0.001, betas=(0.95, 0.99))

    recipe = tch.recipes.TrainAndCall(
        m,
        train,
        test,
        data_loader,
        test_every=200,
        visdom_env=f'vermiffusion_{LR}_{tag}' if rank == 0 else None)

    recipe.callbacks.add_callbacks([
        #tcb.LRSched(tch.lr_scheduler.CosineDecay(opt, len(data_loader) * EPOCHS, warmup_ratio=0.05), metric=None, step_each_batch=True),
        tcb.Optimizer(opt, log_lr=True),
        tcb.Log('loss', 'loss'),
        tcb.Log('x0', 'pred_x0'),
        #tcb.Log('xt', 'xt'),
        #tcb.Log('batch.0', 'batch'),
    ])
    recipe.test_loop.callbacks.add_callbacks([
        tcb.Log('gen', 'gen'),
    ])
    #recipe.load_state_dict(torch.load('model/ckpt_27000.pth', map_location='cpu'))
    print(sum(p.numel() for p in basem.parameters()) / 1e6, 'M parameters')
    recipe.to(rank)
    recipe.run(EPOCHS)


if __name__ == '__main__':
    import sys
    tag = sys.argv[1]
    tu.parallel_run(main, tag)
