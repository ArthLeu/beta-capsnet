from torch.nn.modules.module import Module
from functions.nnd import NNDFunction

class NNDModule(Module):
    @staticmethod
    def forward(input1, input2):
        return NNDFunction()(input1, input2)
