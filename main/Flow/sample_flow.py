import torch
from utils import gen_inn

BATCHSIZE = 1
CAP_DIM = 64
N_CAP = 64
N_DIM = CAP_DIM*N_CAP
model_path = "main/Flow/flow_airplane.pt"

inn = gen_inn(N_DIM)
inn.load_state_dict(torch.load(model_path))
inn.eval()

z = torch.randn(BATCHSIZE, N_DIM)
samples, _ = inn(z, rev=True)
samples = torch.reshape(samples, (1, CAP_DIM, N_CAP))
print(samples)
torch.save(samples.to("cuda:0"), "tmp_lcs/generated_capsules.pt")