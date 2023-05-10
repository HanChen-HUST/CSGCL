from typing import Any, Optional, Callable
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv


class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: Callable,
                 base_model: Any = GCNConv,
                 k: int = 2,
                 skip: bool = False):

        super(Encoder, self).__init__()
        self.base_model = base_model
        assert k >= 2
        self.k = k 
        self.skip = skip
        if not self.skip:
            self.conv = [base_model(in_channels, 2 * out_channels).jittable()]
            for _ in range(1, k - 1):
                self.conv.append(base_model(2 * out_channels, 2 * out_channels))
            self.conv.append(base_model(2 * out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)
            self.activation = activation
        else:
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = [base_model(in_channels, out_channels)]
            for _ in range(1, k):
                self.conv.append(base_model(out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        if not self.skip:
            for i in range(self.k):
                x = self.activation(self.conv[i](x, edge_index))
            return x
        else:
            h = self.activation(self.conv[0](x, edge_index))
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                hs.append(self.activation(self.conv[i](u, edge_index)))
            return hs[-1]


class CSGCL(torch.nn.Module):
    def __init__(self,
                 encoder: Encoder,
                 num_hidden: int,
                 num_proj_hidden: int,
                 tau: float = 0.5):

        super(CSGCL, self).__init__()
        self.encoder = encoder
        self.tau = tau
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
        self.num_hidden = num_hidden

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self,
                   z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def _sim(self,
             z1: torch.Tensor,
             z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def _infonce(self,
                  z1: torch.Tensor,
                  z2: torch.Tensor) -> torch.Tensor:

        temp = lambda x: torch.exp(x / self.tau)
        refl_sim = temp(self._sim(z1, z1))
        between_sim = temp(self._sim(z1, z2))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def _batched_infonce(self,
                          z1: torch.Tensor,
                          z2: torch.Tensor,
                          batch_size: int) -> torch.Tensor:
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []
        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self._sim(z1[mask], z1))
            between_sim = f(self._sim(z1[mask], z2))
            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
        return torch.cat(losses)
        
    def _team_up(self,
                 z1: torch.Tensor,
                 z2: torch.Tensor,
                 cs: torch.Tensor,
                 current_ep: int,
                 t0: int,
                 gamma_max: int) -> torch.Tensor:
        gamma = min(max(0, (current_ep - t0) / 100), gamma_max)
        temp = lambda x: torch.exp(x / self.tau)
        refl_sim = temp(self._sim(z1, z1) + gamma * cs + gamma * cs.unsqueeze(dim=1))
        between_sim = temp(self._sim(z1, z2) + gamma * cs + gamma * cs.unsqueeze(dim=1))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def _batched_team_up(self,
                         z1: torch.Tensor,
                         z2: torch.Tensor,
                         cs: torch.Tensor,
                         current_ep: int,
                         t0: int,
                         gamma_max: int,
                         batch_size: int) -> torch.Tensor:
        gamma = min(max(0, (current_ep - t0) / 100), gamma_max)
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        temp = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = temp(self._sim(z1[mask], z1) + gamma * cs + gamma * cs.unsqueeze(dim=1)[mask])
            between_sim = temp(self._sim(z1[mask], z2) + gamma * cs + gamma * cs.unsqueeze(dim=1)[mask])

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def infonce(self,
                z1: torch.Tensor,
                z2: torch.Tensor,
                mean: bool = True,
                batch_size: Optional[int] = None) -> torch.Tensor:
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size is None:
            l1 = self._infonce(h1, h2)
            l2 = self._infonce(h2, h1)
        else:
            l1 = self._batched_infonce(h1, h2, batch_size)
            l2 = self._batched_infonce(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    def team_up_loss(self,
                     z1: torch.Tensor,
                     z2: torch.Tensor,
                     cs: np.ndarray,
                     current_ep: int,
                     t0: int = 0,
                     gamma_max: int = 1,
                     mean: bool = True,
                     batch_size: Optional[int] = None) -> torch.Tensor:

        h1 = self.projection(z1)
        h2 = self.projection(z2)
        cs = torch.from_numpy(cs).to(h1.device)
        if batch_size is None:
            l1 = self._team_up(h1, h2, cs, current_ep, t0, gamma_max)
            l2 = self._team_up(h2, h1, cs, current_ep, t0, gamma_max)
        else:
            l1 = self._batched_team_up(h1, h2, cs, current_ep, t0, gamma_max, batch_size)
            l2 = self._batched_team_up(h2, h1, cs, current_ep, t0, gamma_max, batch_size)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
