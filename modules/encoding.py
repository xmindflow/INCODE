import math
import numpy as np
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FrequencyEncoding(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, pos_encode_configs, in_features=2):
        super().__init__()

        mapping_input = pos_encode_configs['mapping_input']
        use_nyquist = pos_encode_configs['use_nyquist']
        self.in_features = in_features

        if self.in_features == 3:
            self.num_frequencies = 10
        elif self.in_features == 2:
            assert mapping_input is not None
            if isinstance(mapping_input, int):
                mapping_input = (mapping_input, mapping_input)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(min(mapping_input[0], mapping_input[1]))
        elif self.in_features == 1:
            #assert fn_samples is not None
            fn_samples = mapping_input
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)
        else:
            self.num_frequencies = 4

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)


class GaussianEncoding(nn.Module):
    def __init__(self, pos_encode_configs, in_features=2):
        super().__init__()
        
        self.scale = pos_encode_configs['scale_B']
        mapping_input = pos_encode_configs['mapping_input']
        
        self.B_gauss = torch.randn((mapping_input, in_features), device=device) * self.scale
        self.out_dim = mapping_input * in_features

    def forward(self, coords):
        x_proj = (2. * np.pi * coords) @ self.B_gauss.t()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Encoding(nn.Module):
    def __init__(self, encoding=None):
        self.encoding_dict = {'frequency': FrequencyEncoding,
                              'gaussian': GaussianEncoding
                             }
        if encoding != None:
            self.encoding = self.encoding_dict[encoding]
        
    def run(self, *args, **kwargs):
        return self.encoding(*args, **kwargs)