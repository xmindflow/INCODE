import numpy as np
import torch
from torch import nn
from modules.encoding import Encoding

    
class GaborLayer(nn.Module):
    """
    GaborLayer and GaborNet from https://github.com/addy1997/mfn-pytorch/blob/main/model/MultiplicativeFilterNetworks.py

    Args:
        in_dim (int): Number of input features.
        out_dim (int): Number of output features.
        padding (int): Padding size for the Gabor filters.
        alpha (float): Shape parameter for the Gamma distribution of Gabor filter scales.
        beta (float, optional): Scale parameter for the Gamma distribution of Gabor filter scales. Default is 1.0.
        bias (bool, optional): If True, enables bias for the linear operation. Default is False.

    """
    def __init__(self, in_dim, out_dim, padding, alpha, beta=1.0, bias=False):
        super(GaborLayer, self).__init__()

        self.mu = nn.Parameter(torch.rand((out_dim, in_dim)) * 2 - 1)
        self.gamma = nn.Parameter(torch.distributions.gamma.Gamma(alpha, beta).sample((out_dim, )))
        self.linear = torch.nn.Linear(in_dim, out_dim)
        #self.padding = padding

        self.linear.weight.data *= 128. * torch.sqrt(self.gamma.unsqueeze(-1))
        self.linear.bias.data.uniform_(-np.pi, np.pi)

        # Bias parameters start in zeros
        #self.bias = nn.Parameter(torch.zeros(self.responses)) if bias else None

    def forward(self, input):
        norm = (input ** 2).sum(dim=1).unsqueeze(-1) + (self.mu ** 2).sum(dim=1).unsqueeze(0) - 2 * input @ self.mu.T
        return torch.exp(- self.gamma.unsqueeze(0) / 2. * norm) * torch.sin(self.linear(input))


class INR(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True,
                 pos_encode_configs={'type':None, 'use_nyquist': None, 'scale_B': None, 'mapping_input': None}):
        super(INR, self).__init__()

        # Positional Encoding
        self.pos_encode = pos_encode_configs['type']
        if self.pos_encode in Encoding().encoding_dict.keys():
            self.positional_encoding = Encoding(self.pos_encode).run(in_features=in_features, pos_encode_configs=pos_encode_configs)
            in_features = self.positional_encoding.out_dim
        elif self.pos_encode == None: 
            self.pos_encode = False
        else:
            assert "Invalid pos_encode. Choose from: [frequency, Gaussian]"


        self.k = hidden_layers+1
        self.gabon_filters = nn.ModuleList([GaborLayer(in_features, hidden_features, 0, alpha=6.0 / self.k) for _ in range(self.k)])
        self.linear = nn.ModuleList(
            [torch.nn.Linear(hidden_features, hidden_features) for _ in range(self.k - 1)] + [torch.nn.Linear(hidden_features, out_features)])

        for lin in self.linear[:self.k - 1]:
            lin.weight.data.uniform_(-np.sqrt(1.0 / hidden_features), np.sqrt(1.0 / hidden_features))

    def forward(self, coords):
        
        if self.pos_encode:
            coords = self.positional_encoding(coords)

        # Recursion - Equation 3
        zi = self.gabon_filters[0](coords[0, ...])  # Eq 3.a
        for i in range(self.k - 1):
            zi = self.linear[i](zi) * self.gabon_filters[i + 1](coords[0, ...])
            # Eq 3.b

        return self.linear[self.k - 1](zi)[None, ...]  # Eq 3.c