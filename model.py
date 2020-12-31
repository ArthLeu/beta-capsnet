"""bcaps_model.py"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

import sys, os

# function definitions

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


# aux modules

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

# encoding layers

class ConvLayer(nn.Module):
    def __init__(self):
        super(ConvLayer, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1) ###
        self.conv2 = torch.nn.Conv1d(64, 128, 1) ###
        self.bn1 = nn.BatchNorm1d(64) ##
        self.bn2 = nn.BatchNorm1d(128) ##

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))) ###
        x = F.relu(self.bn2(self.conv2(x))) ###
        return x


class PrimaryPointCapsLayer(nn.Module):
    def __init__(self, prim_vec_size=8, num_points=2048):
        super(PrimaryPointCapsLayer, self).__init__()
        self.capsules = nn.ModuleList([
            torch.nn.Sequential(OrderedDict([
                ('conv3', torch.nn.Conv1d(128, 1024, 1)),
                ('bn3', nn.BatchNorm1d(1024)),
                ('mp1', torch.nn.MaxPool1d(num_points)),
            ]))
            for _ in range(prim_vec_size)])
    
    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=2)
        return self.squash(u.squeeze())
    
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / \
            ((1. + squared_norm) * torch.sqrt(squared_norm))
        if(output_tensor.dim() == 2):
            output_tensor = torch.unsqueeze(output_tensor, 0)
        return output_tensor


class LatentCapsLayer(nn.Module):
    def __init__(self, latent_caps_size=16, prim_caps_size=1024, prim_vec_size=16, latent_vec_size=64):
        super(LatentCapsLayer, self).__init__()
        self.prim_vec_size = prim_vec_size
        self.prim_caps_size = prim_caps_size
        self.latent_caps_size = latent_caps_size
        self.W = nn.Parameter(0.01*torch.randn(latent_caps_size, prim_caps_size, latent_vec_size, prim_vec_size))

    def forward(self, x): # https://pechyonkin.me/capsules-3/
        u_hat = torch.squeeze(torch.matmul(self.W, x[:, None, :, :, None]), dim=-1)
        u_hat_detached = u_hat.detach()
        b_ij = Variable(torch.zeros(x.size(0), self.latent_caps_size, self.prim_caps_size)).cuda()
        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, 1)
            if iteration == num_iterations - 1:
                v_j = self.squash(torch.sum(c_ij[:, :, :, None] * u_hat, dim=-2, keepdim=True))
            else:
                v_j = self.squash(torch.sum(c_ij[:, :, :, None] * u_hat_detached, dim=-2, keepdim=True))
                b_ij = b_ij + torch.sum(v_j * u_hat_detached, dim=-1)
        return v_j.squeeze(-2) # removes given dimension if size is 1
    
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / \
            ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


# decoding layers


class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=2500):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, int(self.bottleneck_size/2), 1)
        self.conv3 = torch.nn.Conv1d(int(self.bottleneck_size/2), int(self.bottleneck_size/4), 1)
        self.conv4 = torch.nn.Conv1d(int(self.bottleneck_size/4), 3, 1)
        self.th = torch.nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(int(self.bottleneck_size/2))
        self.bn3 = torch.nn.BatchNorm1d(int(self.bottleneck_size/4))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x


class CapsDecoder(nn.Module):
    def __init__(self, latent_caps_size, latent_vec_size, num_points):
        super(CapsDecoder, self).__init__()
        self.latent_caps_size = latent_caps_size
        self.bottleneck_size=latent_vec_size
        self.num_points = num_points
        self.nb_primitives=int(num_points/latent_caps_size)
        self.decoder = nn.ModuleList(
            [PointGenCon(bottleneck_size=self.bottleneck_size+2) for i in range(0, self.nb_primitives)])
    def forward(self, x):
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2, self.latent_caps_size))
            rand_grid.data.uniform_(0, 1)
            #print(x.size())
            #print(rand_grid.size())
            y = torch.cat((rand_grid, x.transpose(2, 1)), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous()


# entirety

class BetaPointCapsNet(nn.Module):
    """Model combining beta-VAE with 3D Capsnet."""

    def __init__(self, prim_caps_size, prim_vec_size, latent_caps_size, latent_vec_size, num_points):
        super(BetaPointCapsNet, self).__init__()
        self.latent_caps_size = latent_caps_size
        # encoder
        self.conv_layer = ConvLayer()
        self.primary_point_caps_layer = PrimaryPointCapsLayer(prim_vec_size, num_points)
        #self.latent_caps_layer = LatentCapsLayer(latent_caps_size * 2, prim_caps_size, prim_vec_size, latent_vec_size) # times 2 since we need mu and logvar for VAE
        self.latent_caps_layer = LatentCapsLayer(latent_caps_size, prim_caps_size, prim_vec_size, latent_vec_size)
        self.Z = CapsuleVAE(latent_vec_size)
        # decoder
        self.caps_decoder = CapsDecoder(latent_caps_size,latent_vec_size, num_points)

    def forward(self, x):
        latent_caps, caps_recon, logvar = self._encode(x) # distribution makes sense when capsule vector size is 1
        x_recon = self._decode(caps_recon).view(x.size()) # x_reconstructed
        return (x_recon, latent_caps, caps_recon, logvar)

    def _encode(self, x):
        x1 = self.conv_layer(x)
        x2 = self.primary_point_caps_layer(x1)
        latent_caps = self.latent_caps_layer(x2)
        caps_recon, logvar = self.Z(latent_caps)
        return latent_caps, caps_recon, logvar

    def _decode(self, caps_recon):
        reconstructions = self.caps_decoder(caps_recon)
        return reconstructions


class PointCapsNet(nn.Module):
    ''' original Point Capsnet by Zhao et. al.'''
    def __init__(self, prim_caps_size, prim_vec_size, latent_caps_size, latent_vec_size, num_points):
        super(PointCapsNet, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_point_caps_layer = PrimaryPointCapsLayer(prim_vec_size, num_points)
        self.latent_caps_layer = LatentCapsLayer(latent_caps_size, prim_caps_size, prim_vec_size, latent_vec_size)
        self.caps_decoder = CapsDecoder(latent_caps_size,latent_vec_size, num_points)

    def forward(self, data):
        x1 = self.conv_layer(data)
        x2 = self.primary_point_caps_layer(x1)
        latent_capsules = self.latent_caps_layer(x2)
        reconstructions = self.caps_decoder(latent_capsules)
        return latent_capsules, reconstructions
        

class CapsuleBVAE(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, latent_vec_size):
        super(CapsuleBVAE, self).__init__()
        self.z_dim = latent_vec_size
        self.fc_mu = nn.Linear(latent_vec_size, self.z_dim)
        self.fc_var = nn.Linear(latent_vec_size, self.z_dim)

        self.decoder = nn.Linear(self.z_dim, latent_vec_size)

        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            kaiming_init(m)
            #for m in self._modules[block]:
            #    kaiming_init(m)

    def forward(self, caps):
        #mu = x
        mu = self.fc_mu(caps)
        logvar = self.fc_var(caps)
        z = reparametrize(mu, logvar)
        
        caps_recon = self._decode(z) # x_reconstructed

        return caps_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)



if __name__ == '__main__':

    from chamfer_distance import ChamferDistance
    CD = ChamferDistance()

    USE_CUDA = True
    batch_size = 2 # ORIGINAL WAS 8
    
    prim_caps_size = 1024
    prim_vec_size = 16
    
    latent_caps_size = 32 # number of latent capsules
    latent_vec_size = 16  # scale of (number of neurons in) latent capsules
    
    num_points = 2048

    point_caps_ae = BetaPointCapsNet(prim_caps_size,prim_vec_size,latent_caps_size,latent_vec_size,num_points)
    point_caps_ae=torch.nn.DataParallel(point_caps_ae).cuda()
    
    rand_data = torch.rand(batch_size,num_points, 3) 
    rand_data = Variable(rand_data)
    rand_data = rand_data.transpose(2, 1)
    rand_data = rand_data.cuda()
    
    reconstruction = point_caps_ae(rand_data)[0] # what forward() function returns, e.g. x_recon, mu, logvar

    rand_data_ = rand_data.transpose(2, 1).contiguous()
    reconstruction_ = reconstruction.transpose(2, 1).contiguous()
    dist1, dist2 = CD(rand_data_, reconstruction_)
    loss = (torch.mean(dist1)) + (torch.mean(dist2))
    print("[DONE] loss: ",loss.item()) 

    
