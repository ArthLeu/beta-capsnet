import torch
from utils import DataLoader, gen_inn

BATCHSIZE = 20
N_DIM = 64*64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inn = gen_inn(N_DIM)
inn.to(device)

dl = DataLoader(device)
optimizer = torch.optim.Adam(inn.parameters(), lr=0.001)

for i in range(1000):
    optimizer.zero_grad()

    x = dl.next_batch(BATCHSIZE)

    # pass to INN and get transformed variable z and log Jacobian determinant
    z, log_jac_det = inn(x)
    # calculate the negative log-likelihood of the model with a standard normal prior
    loss = 0.5*torch.sum(z**2, 1) - log_jac_det
    loss = loss.mean() / N_DIM
    # backpropagate and update the weights
    loss.backward()
    optimizer.step()

    print(i, loss.item(), end='\r')

torch.save(inn.state_dict(), 'capsule_flow.pt')