import torch
from torch import nn
from torch.nn import MSELoss

import utils
from typing import List


class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, pred, targets):
        diff_sq = (pred - targets) ** 2 / targets.shape[0]
        return torch.sum(self.weights * diff_sq)


class HiggsLoss(nn.Module):
    def __init__(self, targets: List[str], alphas: List[float] = None, output_mean=None, output_std=None, target_mean=None, target_std=None):
        super().__init__()
        self.targets = targets
        _, self.output_transform, self.target_transform = utils.init_transforms(
            fit_transforms=False, output_mean=output_mean, output_std=output_std, target_mean=target_mean, target_std=target_std)
        if alphas is not None:
            self.loss = WeightedMSELoss(alphas)
        else:
            self.loss = MSELoss()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, attributes: torch.Tensor):
        self.output_transform.to(outputs.device)
        self.target_transform.to(outputs.device)

        breakpoint()
        outputs = self.output_transform.inverse_transform(outputs)
        Nbx_pred, Nby_pred, Wam_pred, Wbm_pred, Hm_pred = calc_tree(
            outputs, attributes)

        pred_map = {'Na_Genx': outputs[:, 0], 'Na_Geny': outputs[:, 1], 'Na_Genz': outputs[:, 2], 'Nb_Genx': Nbx_pred,
                    'Nb_Geny': Nby_pred, 'Nb_Genz': outputs[:, 3], 'Wa_Genm': Wam_pred, 'Wb_Genm': Wbm_pred, 'H_Genm': Hm_pred}
        pred = torch.stack([pred_map[t] for t in self.targets], dim=1)
        pred = self.target_transform(pred)
        return self.loss(pred, targets)


def calc_tree(outputs: torch.Tensor, attributes: torch.Tensor):
    # Depends on order of outputs and attributes in dataset config.
    METx = attributes[:, 0]
    METy = attributes[:, 1]
    Lax_vis = attributes[:, 2]
    Lay_vis = attributes[:, 3]
    Laz_vis = attributes[:, 4]
    Lam_vis = attributes[:, 5]
    Lbx_vis = attributes[:, 6]
    Lby_vis = attributes[:, 7]
    Lbz_vis = attributes[:, 8]
    Lbm_vis = attributes[:, 9]
    Nax_pred = outputs[:, 0]
    Nay_pred = outputs[:, 1]
    Naz_pred = outputs[:, 2]
    Nam_pred = torch.zeros_like(Nax_pred)  # Approximate 0 neutrino mass.
    Nbx_pred = METx - Nax_pred
    Nby_pred = METy - Nay_pred
    Nbz_pred = outputs[:, 3]
    Nbm_pred = torch.zeros_like(Nbx_pred)  # Approximate 0 neutrino mass.
    Wax_pred, Way_pred, Waz_pred, Wam_pred = utils.add_fourvectors(
        Nax_pred, Nay_pred, Naz_pred, Nam_pred, Lax_vis, Lay_vis, Laz_vis, Lam_vis)
    Wbx_pred, Wby_pred, Wbz_pred, Wbm_pred = utils.add_fourvectors(
        Nbx_pred, Nby_pred, Nbz_pred, Nbm_pred, Lbx_vis, Lby_vis, Lbz_vis, Lbm_vis)
    _, _, _, Hm_pred = utils.add_fourvectors(Wax_pred, Way_pred, Waz_pred, Wam_pred,
                                             Wbx_pred, Wby_pred, Wbz_pred, Wbm_pred)
    return Nbx_pred, Nby_pred, Wam_pred, Wbm_pred, Hm_pred
