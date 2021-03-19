## TODO: build an NN class with input latent caps output latent caps
"""train_capsule_bvae.py"""

import argparse
import sys
import os
import glob

import torch
import torch.nn.parallel
from torch.autograd import Variable
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../../')))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../AE/')))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../../dataloaders')))

import shapenet_part_loader
import shapenet_core13_loader
import shapenet_core55_loader
from model import CapsuleBVAE
from solver import kl_divergence, reconstruction_loss
from logger import Logger

USE_CUDA = True
LOGGING = False
CLASS = "Airplane"

def load_batch_latent_from_file(file_list):
    batch_size = opt.batch_size
    latent_caps_size = opt.latent_caps_size
    latent_vec_size = opt.latent_vec_size

    all_batch = []

    for i in range(len(file_list)//batch_size):

        latent_capsules = torch.zeros([batch_size, latent_caps_size, latent_vec_size])

        for j in range(batch_size):
            filepath = file_list[i * batch_size + j]
            lc = torch.load(filepath)
            latent_capsules[j, :] = lc
        
        all_batch.append(latent_capsules.unsqueeze(1))

    return all_batch # in batch

def load_all_latents(file_list, batch_size):
    all_latents = []
    for f in file_list:
        lc = torch.load(f)
        lc = lc.unsqueeze(0)
        all_latents.append(lc)
    D = torch.utils.data.DataLoader(all_latents, batch_size=batch_size)
    return D


def main():
    USE_CUDA = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cbvae = CapsuleBVAE(z_dim = opt.z_dim)
  
    if opt.model != '':
        cbvae.load_state_dict(torch.load(opt.model))
 
    if USE_CUDA:       
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        cbvae = torch.nn.DataParallel(cbvae)
        cbvae.to(device)

    # create folder to save trained models
    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)

    # create folder to save logs
    if LOGGING:
        log_dir='./logs'+'/'+opt.dataset+'_bvae_dataset_'+str(opt.latent_caps_size)+'caps_'+str(opt.latent_vec_size)+'vec'+'_batch_size_'+str(opt.batch_size)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logger = Logger(log_dir)

    base_path = "/mnt/massdisk/dataset/latent_capsules/%s/*.pt"%(CLASS.lower())
    file_list = glob.glob(base_path)
    print("[INFO] Found %d latent capsules"%(len(file_list)))
    all_batch = load_all_latents(file_list, opt.batch_size)
    #all_batch = load_batch_latent_from_file(file_list)
    #print("[INFO] Loaded %d batches in total"%(all_batch))

    # BVAE CONFIGURATIONS HARDCODING
    loss_mode = 'gaussian' # loss_mode was decoder_list in bVAE
    #loss_mode = 'chamfer' 

    loss_objective = "H" # Higgin et al "H", or Burgess et al "B"

    C_max = 25          # default 25, pending addition to args
    C_stop_iter = 1e5   # default 1e5, pending addition to args
    global_iter = 0     # iteration count
    C_max = Variable(torch.FloatTensor([C_max]).cuda()) # use_cuda = True

    gamma = 1000        # default 1000, pending addition to args
    beta = 0.8            # default 4, pending addition to args


    cbvae.train()
    for epoch in range(opt.n_epochs+1):
        if epoch < 50:
            optimizer = optim.Adam(cbvae.parameters(), lr=0.01)
        elif epoch<150:
            optimizer = optim.Adam(cbvae.parameters(), lr=0.001)
        else:
            optimizer = optim.Adam(cbvae.parameters(), lr=0.0001)

        
        train_loss_sum, recon_loss_sum, beta_loss_sum = 0, 0, 0

        for batch_id, latent_capsules in enumerate(all_batch):
            if latent_capsules.size(0) < opt.batch_size:
                break

            global_iter += 1

            if USE_CUDA:
                latent_capsules = latent_capsules.cuda()

            optimizer.zero_grad()
            
            # ---- CRITICAL PART: new train loss computation (train_loss in bVAE was beta_vae_loss)
            
            caps_recon, mu, logvar = cbvae(latent_capsules)
            recon_loss = reconstruction_loss(latent_capsules, caps_recon, "mse")
            total_kld, _, _ = kl_divergence(mu, logvar) # DIVERGENCE

            if loss_objective == 'H':
                beta_loss = beta * total_kld
            elif loss_objective == 'B':
                C = torch.clamp(C_max/C_stop_iter*global_iter, 0, C_max.data[0])
                beta_loss = gamma*(total_kld-C).abs()

            # sum of losses
            beta_total_loss = beta_loss.sum()
            train_loss = recon_loss + beta_loss # LOSS (can be weighted)
            
            # original train loss computation
            #train_loss = capsule_net.module.loss(points, x_recon)
            #train_loss = recon_loss
            #train_loss.backward()

            # combining per capsule loss (pyTorch requires)
            train_loss.backward(retain_graph=True)
            optimizer.step()
            train_loss_sum += train_loss.item()

            # ---- END OF CRITICAL PART ----
            
            if LOGGING:
                info = {'train loss': train_loss.item()}
                for tag, value in info.items():
                    logger.scalar_summary(
                        tag, value, (len(all_batch) * epoch) + batch_id + 1)                
            
            if batch_id % 5 == 0:
                print('batch_no: %d / %d, train_loss: %f ' %  (batch_id, len(all_batch), train_loss.item()))

        print('\nAverage train loss of epoch %d : %f\n' %\
            (epoch, (train_loss_sum / len(all_batch))))

        if epoch% 10 == 0:
            dict_name = "%s/capsbvae_%s_%d.pth"%\
                (opt.outf, CLASS.lower(), epoch)
            torch.save(cbvae.module.state_dict(), dict_name)

            torch.save(mu, "%s/%s_latent_mu.pt"%(opt.outf, CLASS.lower()))
            torch.save(logvar, "%s/%s_latent_var.pt"%(opt.outf, CLASS.lower() ) )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs to train for')

    parser.add_argument('--z_dim', type=int, default=16, help='secondary latent space dimension z')
    parser.add_argument('--latent_caps_size', type=int, default=64, help='total number of latent capsules')
    parser.add_argument('--latent_vec_size', type=int, default=64, help='scale of latent caps')

    parser.add_argument('--model', type=str, default="", help='model path')
    parser.add_argument('--save_training', help='save the output latent caps of training data or test data', action='store_true')
    parser.add_argument('--outf', type=str, default="tmp_checkpoints_cbvae", help='checkpoint saving path')
    opt = parser.parse_args()
    print(opt)
    main()