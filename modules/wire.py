import numpy as np
import torch
from torch import nn
from modules.encoding import Encoding


class RealGaborLayer(nn.Module):
    '''
    RealGaborLayer is a custom PyTorch module that applies the Gabor activation function to the output of a linear transformation.
        
    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If True, enable bias for the linear operations. Default is True.
        is_first (bool, optional): Legacy SIREN parameter. Default is False.
        omega0 (float, optional): Frequency scaling factor for the Gabor sinusoid term. Default is 10.0.
        sigma0 (float, optional): Scaling factor for the Gabor Gaussian term. Default is 10.0.
        trainable (bool, optional): If True, allows training of the omega and scale parameters. Default is False.
    '''
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=10.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.in_features = in_features
        
        self.freqs = nn.Linear(in_features, out_features, bias=bias)
        self.scale = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input):
        omega = self.omega_0 * self.freqs(input)
        scale = self.scale(input) * self.scale_0
        
        return torch.cos(omega)*torch.exp(-(scale**2))
    

class ComplexGaborLayer(nn.Module):
    '''
    ComplexGaborLayer is a custom PyTorch module that applies the ComplexGabor activation function to the output of a linear transformation.
        
    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If True, enable bias for the linear operations. Default is True.
        is_first (bool, optional): Legacy SIREN parameter. Default is False.
        omega0 (float, optional): Frequency scaling factor for the Gabor sinusoid term. Default is 10.0.
        sigma0 (float, optional): Scaling factor for the Gabor Gaussian term. Default is 10.0.
        trainable (bool, optional): If True, allows training of the omega and scale parameters. Default is False.

    '''
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=40.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.in_features = in_features
        
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
            
        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0*torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0*torch.ones(1), trainable)
        
        self.linear = nn.Linear(in_features,
                                out_features,
                                bias=bias,
                                dtype=dtype)
    
    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        
        return torch.exp(1j*omega - scale.abs().square())
    

class INR(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, wire_type='complex',
                 outermost_linear=True, first_omega_0=30, hidden_omega_0=30, sigma=10.0,
                 pos_encode_configs={'type':None, 'use_nyquist': None, 'scale_B': None, 'mapping_input': None}):
        super().__init__()
        
        # Positional Encoding
        self.pos_encode = pos_encode_configs['type']
        if self.pos_encode in Encoding().encoding_dict.keys():
            self.positional_encoding = Encoding(self.pos_encode).run(in_features=in_features, pos_encode_configs=pos_encode_configs)
            in_features = self.positional_encoding.out_dim
        elif self.pos_encode == None: 
            self.pos_encode = False
        else:
            assert "Invalid pos_encode. Choose from: [frequency, Gaussian]"


        if wire_type == 'real':
            self.nonlin = RealGaborLayer
        else:
            self.nonlin = ComplexGaborLayer
        
        # Since complex numbers are two real numbers, reduce the number of hidden parameters by 2
        hidden_features = int(hidden_features/np.sqrt(2))
        dtype = torch.cfloat
        self.complex = True
        self.wavelet = 'gabor'    
        
        # Legacy parameter
        self.pos_encode = False
            
        self.net = []
        self.net.append(self.nonlin(in_features,
                                    hidden_features, 
                                    omega0=first_omega_0,
                                    sigma0=sigma,
                                    is_first=True,
                                    trainable=False))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features,
                                        hidden_features, 
                                        omega0=hidden_omega_0,
                                        sigma0=sigma))

        final_linear = nn.Linear(hidden_features,
                                 out_features,
                                 dtype=dtype)            
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        if self.pos_encode:
            coords = self.positional_encoding(coords)

        output = self.net(coords)
        
        if self.wavelet == 'gabor':
            return output.real
         
        return output