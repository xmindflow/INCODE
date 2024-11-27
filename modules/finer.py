import numpy as np
import torch
from torch import nn
from modules.encoding import Encoding



class FinerLayer(nn.Module):
    """
    FinerLayer is a custom PyTorch module that applies a linear transformation followed by a scaled sinusoidal activation.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        bias (bool, optional): If True, adds a learnable bias to the output. Default is True.
        is_first (bool, optional): If True, this layer is the first in the network, and weights are initialized differently. Default is False.
        omega_0 (float, optional): Frequency factor for the sine activation. Default is 30.
        first_bias_scale (float, optional): Scale for initializing the bias of the first layer. Only used if `is_first` is True. Default is None.
        scale_req_grad (bool, optional): If True, the scale factor is computed with gradient tracking. Default is False.
    """
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30, first_bias_scale=None, scale_req_grad=False):
        super().__init__()
        # Store parameters
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.scale_req_grad = scale_req_grad
        self.first_bias_scale = first_bias_scale

        # Define a linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Initialize weights for the layer
        self.init_weights()

        # If first_bias_scale is provided, initialize the bias for the first layer
        if self.first_bias_scale is not None:
            self.init_first_bias()
    
    def init_weights(self):
        """
        Initialize weights for the linear layer. The initialization differs depending on whether
        this layer is the first in the network or not.
        """
        with torch.no_grad():
            # Initialize weights differently for the first layer and other layers
            if self.is_first:
                # For the first layer, initialize weights uniformly between -1/in_features and 1/in_features
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                # For other layers, initialize weights using a scaled uniform distribution
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def init_first_bias(self):
        """
        Initialize the bias for the first layer if `first_bias_scale` is provided.
        """
        with torch.no_grad():
            # Initialize the bias of the first layer with a uniform distribution between -first_bias_scale and first_bias_scale
            if self.is_first:
                self.linear.bias.uniform_(-self.first_bias_scale, self.first_bias_scale)

    def generate_scale(self, x):
        """
        Generate a scale factor based on the magnitude of the input tensor plus one.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The scale factor tensor.
        """
        # Generate a scale factor based on the magnitude of x plus one
        if self.scale_req_grad: 
            # If scale_req_grad is True, compute scale with gradient tracking
            scale = torch.abs(x) + 1
        else:
            # If scale_req_grad is False, compute scale without gradient tracking
            with torch.no_grad():
                scale = torch.abs(x) + 1
        return scale
        
    def forward(self, input):
        """
        Forward pass of the FinerLayer.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying the linear transformation and sinusoidal activation.
        """
        # Apply the linear transformation to the input
        x = self.linear(input)
        
        # Generate a scale factor from the output of the linear layer
        scale = self.generate_scale(x)
        
        # Apply a sinusoidal activation function, scaled by omega_0 and the generated scale
        out = torch.sin(self.omega_0 * scale * x)
        return out
    
    
class INR(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, ffn_type=None,
                 outermost_linear=True, first_omega_0=30, hidden_omega_0=30, first_bias_scale=None, scale_req_grad=False,
                 pos_encode_configs={'type':None, 'use_nyquist': None, 'scale_B': None, 'mapping_input': None}):
        super().__init__()


        # Positional Encoding        
        if ffn_type == 'finer':
            self.pos_encode = 'gaussian'
        else:
            self.pos_encode = pos_encode_configs['type']
        
        if self.pos_encode in Encoding().encoding_dict.keys():
            self.positional_encoding = Encoding(self.pos_encode).run(in_features=in_features, pos_encode_configs=pos_encode_configs)
            in_features = self.positional_encoding.out_dim
        elif self.pos_encode == None: 
            self.pos_encode = False
        else:
            assert "Invalid pos_encode. Choose from: [frequency, Gaussian]"

        
        
        self.nonlin = FinerLayer    
        self.net = []
        self.net.append(self.nonlin(in_features, hidden_features, 
                                   is_first=True, omega_0=first_omega_0, first_bias_scale=first_bias_scale, scale_req_grad=scale_req_grad))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features, 
                                       is_first=False, omega_0=hidden_omega_0, scale_req_grad=scale_req_grad))

        if outermost_linear:
            dtype = torch.float
            final_linear = nn.Linear(hidden_features,
                                     out_features,
                                     dtype=dtype)
            
            with torch.no_grad():
                const = np.sqrt(6/hidden_features)/max(hidden_omega_0, 1e-12)
                final_linear.weight.uniform_(-const, const)
                    
            self.net.append(final_linear)
        else:
            self.net.append(self.nonlin(hidden_features, out_features, 
                                       is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        if self.pos_encode:
            coords = self.positional_encoding(coords)
            
        output = self.net(coords)
                    
        return output
  