import argparse
import sys
import os

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

from model import PointCapsNet
from open3d import *
import matplotlib.pyplot as plt

from chamfer_distance import ChamferDistance
CD = ChamferDistance()

## MONKEY PATCHING
PointCloud = geometry.PointCloud
Vector3dVector = utility.Vector3dVector
draw_geometries = visualization.draw_geometries
viz = visualization.Visualizer()

image_id = 0
USE_CUDA = True


def show_points(points_tensor):
    print("showing tensor of shape", points_tensor.size())
    prc_r_all=points_tensor.transpose(1, 0).contiguous().data.cpu()
    prc_r_all_point=PointCloud()
    prc_r_all_point.points = Vector3dVector(prc_r_all)
    draw_geometries([prc_r_all_point])

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    capsule_net = PointCapsNet(opt.prim_caps_size, opt.prim_vec_size, opt.latent_caps_size, opt.latent_vec_size, opt.num_points)
  
    if opt.model != '':
        print(opt.model)
        capsule_net.load_state_dict(torch.load(opt.model))
    else:
        print ('pls set the model path')
        
    if USE_CUDA:       
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        capsule_net = torch.nn.DataParallel(capsule_net)
        capsule_net.to(device)

    capsule_net.eval() #CRUICIAL

    for i in range(opt.batch_size):

        latent_filename = "tmp_lcs/cbvae_latcaps_airplane_%03d.pt"%i
        #latent_filename = "/mnt/massdisk/dataset/latent_capsules/airplane/latcaps_airplane_000.pt"
        print(latent_filename)
        slc = torch.load(latent_filename)
        if slc.dim() == 2: slc = slc.unsqueeze(0)
        
        recon1 = capsule_net.module.caps_decoder(slc)
        show_points(recon1[0])


if __name__ == "__main__":
    from open3d import *
    import matplotlib.pyplot as plt
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs to train for')

    parser.add_argument('--prim_caps_size', type=int, default=1024, help='number of primary point caps')
    parser.add_argument('--prim_vec_size', type=int, default=16, help='scale of primary point caps')
    parser.add_argument('--latent_caps_size', type=int, default=64, help='number of latent caps')
    parser.add_argument('--latent_vec_size', type=int, default=64, help='scale of latent caps')

    parser.add_argument('--num_points', type=int, default=2048, help='input point set size')
    parser.add_argument('--model', type=str, default='checkpoints/shapenet_part_dataset_ae_200.pth', help='model path')
    parser.add_argument('--dataset', type=str, default='shapenet_part', help='dataset: shapenet_part, shapenet_core13, shapenet_core55')
    opt = parser.parse_args()
    print(opt)

    main()