import torch
from utils import gen_inn

BATCHSIZE = 1
N_DIM = 64*64

inn = gen_inn(N_DIM)
inn.load_state_dict(torch.load('capsule_flow.pt'))
inn.eval()

z = torch.randn(BATCHSIZE, N_DIM)
samples, _ = inn(z, rev=True)
samples = torch.reshape(samples, (1, 64, 64))
print(samples)