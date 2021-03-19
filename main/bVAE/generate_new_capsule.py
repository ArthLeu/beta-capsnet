import argparse
import sys
import os
import pickle

import torch
import torch.nn.parallel
from torch.autograd import Variable
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../../')))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../../dataloaders')))

import shapenet_part_loader
import shapenet_core13_loader
import shapenet_core55_loader
from model import CapsuleBVAE, reparametrize
from solver import kl_divergence, reconstruction_loss

USE_CUDA = True
LOGGING = True


def main(CLASS="None"):
    if CLASS == "None": exit()

    USE_CUDA = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bvae_net = CapsuleBVAE(z_dim = opt.z_dim)
  
    if opt.model != '':
        bvae_net.load_state_dict(torch.load(opt.model))
    else:
        print ('pls set the model path')
 
    if USE_CUDA:       
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        bvae_net = torch.nn.DataParallel(bvae_net)
        bvae_net.to(device)
    

    #  process for 'shapenet_part' or 'shapenet_core13'
    bvae_net.eval()

    test_loss_sum = 0
    
    mu = torch.load("tmp_checkpoints_cbvae/airplane_latent_mu.pt")
    logvar = torch.load("tmp_checkpoints_cbvae/airplane_latent_var.pt")

    if USE_CUDA:
        mu = mu.cuda()
        logvar = logvar.cuda()

    print(mu.size())

    z = reparametrize(mu, logvar)
    latent_caps = bvae_net.module._decode(z)
    print(z)

    for i in range(opt.batch_size):
        torch.save(latent_caps[i,:], "tmp_lcs/cbvae_latcaps_%s_%03d.pt"%(CLASS.lower(), i))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs to train for')

    parser.add_argument('--z_dim', type=int, default=16, help='secondary latent space dimension z')
    parser.add_argument('--latent_caps_size', type=int, default=64, help='number of latent caps')
    parser.add_argument('--latent_vec_size', type=int, default=64, help='scale of latent caps')

    parser.add_argument('--model', type=str, default="tmp_checkpoints_cbvae/capsbvae_airplane_100.pth", help='model path')
    parser.add_argument('--save_training', help='save the output latent caps of training data or test data', action='store_true')
    opt = parser.parse_args()
    print(opt)

    #f = open("classes.pickle", "rb")
    #d = pickle.load(f)
    #classes = list(d.keys())
    
    #for c in classes:
    #    main(c)
    main("Airplane")

