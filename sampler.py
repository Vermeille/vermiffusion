import torch
from diffusion import ForwardProcess, b, FlowMatching, GradientDescent


class DDIM:

    def __init__(self, forward_process: ForwardProcess):
        self.forward_process = forward_process

    def step(self, xt, pred, current_t, next_t, eta=0):
        a_t = b(self.forward_process.alpha_bar[current_t], 4)
        a_tm1 = b(self.forward_process.alpha_bar[next_t], 4)
        sigma_t = (eta * torch.sqrt(
            (1 - a_tm1) / (1 - a_t)) * torch.sqrt(1 - a_t / a_tm1))

        x0 = self.forward_process.to_x0(xt, pred, current_t)
        pred_e = self.forward_process.to_noise(xt, pred, current_t)

        return (a_tm1.sqrt() * x0 +
                torch.sqrt(1 - a_tm1 - sigma_t**2) * pred_e +
                sigma_t * torch.randn_like(x0))

    @torch.no_grad()
    def sample(self, model, shape, n_steps, eta=0, generator=None):
        device = self.forward_process.alpha.device
        steps = torch.linspace(
            len(self.forward_process.alpha_bar) - 1, 0,
            n_steps).int().to(device)
        xt = torch.randn(*shape, device=device, generator=generator)
        for cur, nxt in zip(steps[:-1], steps[1:]):
            pred = model(xt, cur[None])
            xt = self.step(xt, pred, cur, nxt, eta=eta)
        return xt


class FlowSampler:
    def __init__(self, forward_process: FlowMatching):
        self.forward_process = forward_process

    def step(self, xt, pred, current_t, next_t, base_e, eta=0):
        pred_x0 = self.forward_process.to_x0(xt, pred, current_t)
        next_xt = self.forward_process.step(xt, pred, current_t, next_t)
        return next_xt, pred_x0

    @torch.no_grad()
    def sample(self, model, shape, n_steps, eta=0, generator=None):
        device = next(iter(model.parameters())).device
        steps = torch.linspace(
            self.forward_process.T - 1, 0,
            n_steps).int().to(device)
        xt = torch.randn(*shape, device=device, generator=generator)
        base_e = xt
        xts = []
        for cur, nxt in zip(steps[:-1], steps[1:]):
            print(cur)
            pred = model(xt, cur[None])
            xt, x0 = self.step(xt, pred, cur, nxt, base_e, eta=eta)
            xts.append(x0)
        return xts[-1]#torch.cat(xts, dim=0)


class GradientSampler:
    def __init__(self, forward_process: GradientDescent):
        self.forward_process = forward_process

    def step(self, xt, pred, current_t, next_t, base_e, eta=0):
        next_xt = xt + 0.1 * pred
        return next_xt, next_xt

    @torch.no_grad()
    def sample(self, model, shape, n_steps, eta=0, generator=None):
        device = next(iter(model.parameters())).device
        xt = torch.randn(*shape, device=device, generator=generator)
        base_e = xt
        xts = []
        for _ in range(n_steps):
            pred = model(xt, torch.tensor([0] * shape[0], device=device))
            xt, x0 = self.step(xt, pred, None, None, base_e, eta=eta)
            xts.append(x0)
        return x0#torch.cat(xts, dim=0)
