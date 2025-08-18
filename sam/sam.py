import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, sign="min", rho=0.05, lamb=0.1, adaptive=False, xi=0.3, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, lamb=lamb, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.sign = -1 if sign == 'max' else 1
        self.xi = xi

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm() # equation 2 grad norm, p=2
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                self.state[p]["original_grad"] = p.grad.clone()  #  NEW: Save original parameters and gradients
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        # NEW: combine gradients using paper's objective: ∇[L(w) - λ(max L(w+ε)-L(w))]
        for group in self.param_groups:
            lamb = group["lamb"]
            for p in group["params"]:
                if p.grad is None: continue
                
                # Retrieve stored gradients
                orig_grad = self.state[p]["original_grad"]
                perturb_grad = p.grad
                
                # Compute sharpness-aware gradient
                p.grad = orig_grad + self.sign*lamb*(perturb_grad - orig_grad) # max(-) or min(+)
                # regularization, commented
                # if self.sign == '-1' and self.xi:
                #     p.grad -= self.xi*orig_grad
                
                # Restore original parameters
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups