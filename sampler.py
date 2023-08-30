import torch
from diffusion import ForwardProcess, b


class DDIM:

    def __init__(self, forward_process: ForwardProcess):
        self.forward_process = forward_process

    def step(self, xt, v, current_t, next_t, eta=0):
        a_t = b(self.forward_process.alpha_bar[current_t], 4)
        a_tm1 = b(self.forward_process.alpha_bar[next_t], 4)
        sigma_t = (eta * torch.sqrt(
            (1 - a_tm1) / (1 - a_t)) * torch.sqrt(1 - a_t / a_tm1))

        x0 = self.forward_process.to_x0(xt, v, current_t)
        pred_e = self.forward_process.to_noise(xt, v, current_t)

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
            v = model(xt, cur[None])
            xt = self.step(xt, v, cur, nxt, eta=eta)
        return xt
