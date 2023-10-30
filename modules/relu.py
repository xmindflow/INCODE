import torch
from torch import nn
from modules.encoding import Encoding
    

class SwishLayer(nn.Module):
    '''
    SwishLayer is a custom PyTorch module that applies the Swish
    activation function to the output of a linear transformation.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If True, the linear transformation includes a bias term. Default is True.

    '''
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input):
        x = self.linear(input)
        return x * torch.sigmoid(x)
    

class ReLULayer(nn.Module):
    '''
    ReLULayer is a custom PyTorch module that applies the Rectified Linear Unit (ReLU)
    activation function to the output of a linear transformation.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If True, the linear transformation includes a bias term. Default is True.

    '''
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input):
        return nn.functional.relu(self.linear(input))


class INR(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, ffn_type=None, outermost_linear=True,
                 pos_encode_configs={'type':None, 'use_nyquist': None, 'scale_B': None, 'mapping_input': None}):
        super().__init__()

        
        if ffn_type == 'relu':
            self.pos_encode = 'gaussian'
            self.nonlin = ReLULayer
        elif ffn_type == 'swish':
            self.pos_encode = 'gaussian'
            self.nonlin = SwishLayer
        else:
            self.nonlin = ReLULayer
            self.pos_encode = pos_encode_configs['type']


        # Positional Encoding
        if self.pos_encode in Encoding().encoding_dict.keys():
            self.positional_encoding = Encoding(self.pos_encode).run(in_features=in_features, pos_encode_configs=pos_encode_configs)
            in_features = self.positional_encoding.out_dim
        elif self.pos_encode == None: 
            self.pos_encode = False
        else:
            assert "Invalid pos_encode. Choose from: [frequency, Gaussian]"

        

        self.complex = False
            
        self.net = []
        self.net.append(self.nonlin(in_features, hidden_features))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features))

        if outermost_linear:
            if self.complex:
                dtype = torch.cfloat
            else:
                dtype = torch.float
            final_linear = nn.Linear(hidden_features,
                                     out_features,
                                     dtype=dtype)
                        
            self.net.append(final_linear)
        else:
            self.net.append(self.nonlin(hidden_features, out_features))
        
        
        if ffn_type in ['relu', 'swish']:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        if self.pos_encode:
            coords = self.positional_encoding(coords)

        output = self.net(coords)
                    
        return output