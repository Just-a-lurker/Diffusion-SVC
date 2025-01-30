"""
from https://github.com/yxlllc/DDSP-SVC
MIT License
"""
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


class RectifiedFlowShortCut(nn.Module):
    def __init__(self,
                 velocity_fn,
                 out_dims=128,
                 spec_min=-12,
                 spec_max=2,
                 loss_type='l2'):
        super().__init__()
        self.velocity_fn = velocity_fn
        self.out_dims = out_dims
        self.spec_min = spec_min
        self.spec_max = spec_max
        self.loss_type = loss_type

    def reflow_loss(self, x_1, t, cond, loss_type=None):
        x_0 = torch.randn_like(x_1)
        x_t = x_0 + t[:, None, None, None] * (x_1 - x_0)
        v_pred = self.velocity_fn(x_t, 1000 * t, cond, torch.zeros_like(t))

        if loss_type is None:
            loss_type = self.loss_type
        else:
            loss_type = loss_type

        if loss_type == 'l1':
            loss = (x_1 - x_0 - v_pred).abs().mean()
        elif loss_type == 'l2':
            loss = F.mse_loss(x_1 - x_0, v_pred)
        elif loss_type == 'l2_lognorm':
            weights = 0.398942 / t / (1 - t) * torch.exp(-0.5 * torch.log(t / ( 1 - t)) ** 2)
            loss = torch.mean(weights[:, None, None, None] * F.mse_loss(x_1 - x_0, v_pred, reduction='none'))
        else:
            raise NotImplementedError()

        return loss
    
    def sc_loss(self, x_1, t, d, cond):
        x_0 = torch.randn_like(x_1)
        x_t = x_0 + t[:, None, None, None] * (x_1 - x_0)
        v_t = self.velocity_fn(x_t, 1000 * t, cond, 1000 * d)
        x_t2 = x_t + v_t * d[:, None, None, None]
        v_t2 = self.velocity_fn(x_t2, 1000 * (t + d), cond, 1000 * d)
        v_mean = 0.5 *(v_t + v_t2).detach()
        v_pred = self.velocity_fn(x_t, 1000 * t, cond, 2000 * d)
        loss = F.mse_loss(v_mean, v_pred)
        return loss
        
    def sample_euler(self, x, t, dt, cond, alpha=0.0):
        if alpha == 0:
            x += self.velocity_fn(x, 1000 * t, cond, 1000 * dt) * dt
            t += dt
        else:
            beta = 0.5 * alpha * alpha
            x += ((1 + beta * t) * self.velocity_fn(x, 1000 * t, cond, 1000 * dt) - beta * x) * dt
            x += alpha * torch.sqrt((1 - t) * dt) * torch.randn_like(x)
            t += dt
        return x, t

    def forward(self,
                condition,
                gt_spec=None,
                infer=True,
                infer_step=10,
                method='euler',
                t_start=0.0,
                use_tqdm=True):
        cond = condition.transpose(1, 2)  # [B, H, T]
        b, device = condition.shape[0], condition.device
        if t_start is None:
            t_start = 0.0
        if t_start < 0.0:
            t_start = 0.0
        if t_start > 1.0:
            t_start = 1.0
        if not infer:
            x_1 = self.norm_spec(gt_spec)
            x_1 = x_1.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
            t = t_start + (1.0 - t_start) * torch.rand(b, device=device)
            t_rf = torch.clip(t[b // 4 :], 1e-7, 1-1e-7)
            reflow_loss = self.reflow_loss(x_1[b // 4 :], t_rf, cond[b //4 :])
            t_sc = torch.clip(t[: b // 4], 0, 1-1e-3)
            d = 0.5 * (1.0 - t_sc) * torch.rand(b // 4, device=device)
            sc_loss = self.sc_loss(x_1[: b // 4], t_sc, d, cond[: b // 4].detach())
            return reflow_loss, sc_loss
        else:
            shape = (cond.shape[0], 1, self.out_dims, cond.shape[2])  # [B, 1, M, T]

            # initial condition and step size of the ODE
            if gt_spec is None:
                x = torch.randn(shape, device=device)
                t = torch.full((b,), 0.0, device=device)
                dt = torch.full((b,), 1.0 / infer_step, device=device)
            else:
                norm_spec = self.norm_spec(gt_spec)
                norm_spec = norm_spec.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
                x = t_start * norm_spec + (1 - t_start) * torch.randn(shape, device=device)
                t = torch.full((b,), t_start, device=device)
                dt = torch.full((b,), (1.0 - t_start) / infer_step, device=device)

            if method == 'euler':
                if use_tqdm:
                    for i in tqdm(range(infer_step), desc='sample time step', total=infer_step):
                        x, t = self.sample_euler(x, t, dt, cond)
                else:
                    for i in range(infer_step):
                        x, t = self.sample_euler(x, t, dt, cond)                        
            else:
                raise NotImplementedError(method)
            x = x.squeeze(1).transpose(1, 2)  # [B, T, M]

            return self.denorm_spec(x)

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min
