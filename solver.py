"""bcaps_solver.py"""

import os
import sys

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from chamfer_distance import ChamferDistance # https://github.com/chrdiller/pyTorchChamferDistance
CD = ChamferDistance()


def reconstruction_loss(x, x_recon, mode):
    batch_size = x.size(0)
    assert batch_size != 0

    if mode == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif mode == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    elif mode == 'chamfer':
        x_ = x.transpose(2, 1).contiguous()
        reconstructions_ = x_recon.transpose(2, 1).contiguous()
        dist1, dist2 = CD(x_, reconstructions_) # replaced since old torch_nndistance from yongheng ceased working
        recon_loss = (torch.mean(dist1)) + (torch.mean(dist2))
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld