import torch
import torch.nn as nn
import numpy as np
import FrEIA.framework as Ff
import FrEIA.modules as Fm

class DataLoader(object):
    train_set = {}
    test_set = {}
    start_pos = 0
    train_size = 339

    def __init__(self, device):
        self.device = device

    def next_batch(self, batch_size):
        start = self.start_pos
        self.start_pos = self.start_pos + batch_size
        if self.start_pos >= self.train_size:
            self.start_pos = 0
        end = start+batch_size if (start+batch_size)<self.train_size else self.train_size

        # set all training data to a single batch if batch_size is 0
        if batch_size == 0:
            start = 0
            end = self.train_size + 1

        batch_set = torch.zeros([0, 64*64], device=self.device)
        for i in range(start, end):
            latent_filename = "/home/fz261/Repository/beta-capsnet/dataset/latent_capsules/airplane/latcaps_airplane_%03d.pt"%i
            # print(latent_filename)
            slc = torch.load(latent_filename)
            slc = slc.to(self.device)
            # if slc.dim() == 2: slc = slc.unsqueeze(0)
            slc = torch.reshape(slc, (-1,))
            slc = slc.unsqueeze(0)
            batch_set = torch.cat((batch_set, slc),0)
        return batch_set
    
    def batch_range(self, batch_size):
        if batch_size == 0:
            return 1
        return math.ceil( self.train_size / batch_size )

def gen_inn(N_DIM):
    def subnet_fc(dims_in, dims_out):
        return nn.Sequential(nn.Linear(dims_in, 512), nn.ReLU(),
                             nn.Linear(512,  dims_out))
    inn = Ff.SequenceINN(N_DIM)
    for k in range(8):
        inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=False)
    return inn