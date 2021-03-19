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
from model import PointCapsNet
from solver import kl_divergence, reconstruction_loss

USE_CUDA = True
LOGGING = True


def main(CLASS="None"):
    if CLASS == "None": exit()

    USE_CUDA = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    capsule_net = PointCapsNet(opt.prim_caps_size, opt.prim_vec_size, opt.latent_caps_size, opt.latent_vec_size, opt.num_points)
  
    if opt.model != '':
        capsule_net.load_state_dict(torch.load(opt.model))
    else:
        print ('pls set the model path')
 
    if USE_CUDA:       
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        capsule_net = torch.nn.DataParallel(capsule_net)
        capsule_net.to(device)
    
    if opt.dataset=='shapenet_part':
        if opt.save_training:
            split='train'
        else :
            split='test'            
        dataset = shapenet_part_loader.PartDataset(classification=True, npoints=opt.num_points, split=split, class_choice=CLASS)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)        
    elif opt.dataset=='shapenet_core13':
        dataset = shapenet_core13_loader.ShapeNet(normal=False, npoints=opt.num_points, train=opt.save_training)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    elif opt.dataset=='shapenet_core55':
        dataset = shapenet_core55_loader.Shapnet55Dataset(batch_size=opt.batch_size,npoints=opt.num_points, shuffle=True, train=opt.save_training)
    elif opt.dataset=='modelnet40':
        dataset = modelnet40_loader.ModelNetH5Dataset(batch_size=opt.batch_size, npoints=opt.num_points, shuffle=True, train=opt.save_training)


    #  process for 'shapenet_part' or 'shapenet_core13'
    capsule_net.eval()
    
    count = 0

    if 'dataloader' in locals().keys() :
        test_loss_sum = 0
        for batch_id, data in enumerate(dataloader):
            points, _ = data
            if(points.size(0)<opt.batch_size):
                break
            points = Variable(points)
            points = points.transpose(2, 1)
            if USE_CUDA:
                points = points.cuda()
            latent_caps, _ = capsule_net(points)

            for i in range(opt.batch_size):
                torch.save(latent_caps[i,:], "tmp_lcs/latcaps_%s_%03d.pt"%(CLASS.lower(), count))
                count += 1
                if (count+1) % 50 == 0: print(count+1)
    
    else:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
    parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs to train for')

    parser.add_argument('--prim_caps_size', type=int, default=1024, help='number of primary point caps')
    parser.add_argument('--prim_vec_size', type=int, default=16, help='scale of primary point caps')
    parser.add_argument('--latent_caps_size', type=int, default=64, help='number of latent caps')
    parser.add_argument('--latent_vec_size', type=int, default=64, help='scale of latent caps')

    parser.add_argument('--num_points', type=int, default=2048, help='input point set size')
    parser.add_argument('--model', type=str, default="checkpoints/shapenet_part_dataset_ae_200.pth", help='model path')
    parser.add_argument('--dataset', type=str, default='shapenet_part', help='dataset: shapenet_part, shapenet_core13, shapenet_core55, modelnet40')
    parser.add_argument('--save_training', help='save the output latent caps of training data or test data', action='store_true')
    opt = parser.parse_args()
    print(opt)

    f = open("classes.pickle", "rb")
    d = pickle.load(f)
    classes = list(d.keys())
    
    for c in classes:
        main(c)

