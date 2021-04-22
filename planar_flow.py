
import torch
import torch.nn as nn
from torch import Tensor


class PlanarTransform(nn.Module):

    def __init__(self, dim: int = 2):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1, dim).normal_(0, 0.1))
        self.b = nn.Parameter(torch.randn(1).normal_(0, 0.1))
        self.u = nn.Parameter(torch.randn(1, dim).normal_(0, 0.1))

    def forward(self, z):
        if torch.mm(self.u, self.w.T) < -1:
            self.get_u_hat()

        return z + self.u * nn.Tanh()(torch.mm(z, self.w.T) + self.b)

    def log_det_J(self, z):
        if torch.mm(self.u, self.w.T) < -1:
            self.get_u_hat()
        a = torch.mm(z, self.w.T) + self.b
        psi = (1 - nn.Tanh()(a) ** 2) * self.w
        abs_det = (1 + torch.mm(self.u, psi.T)).abs()
        log_det = torch.log(1e-4 + abs_det)

        return log_det

    def get_u_hat(self):
        wtu = torch.mm(self.u, self.w.T)
        m_wtu = -1 + torch.log(1 + torch.exp(wtu))
        self.u.data = (
            self.u + (m_wtu - wtu) * self.w / torch.norm(self.w, p=2, dim=1) ** 2
        )


class PlanarFlow(nn.Module):
    def __init__(self, dim=2, K=6):
        super().__init__()
        self.layers = [PlanarTransform(dim) for _ in range(K)]
        self.model = nn.Sequential(*self.layers)

    def forward(self, z):
        log_det_J = 0

        for layer in self.layers:
            log_det_J += layer.log_det_J(z)
            z = layer(z)

        return z, log_det_J
