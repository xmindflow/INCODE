import numpy as np
import torch
from torch import nn
from modules.encoding import Encoding
import torchvision.models as models
import torchvision.models.video as video
import torchaudio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(torch.nn.Sequential):
    '''
    Args:
        in_channels (int): Number of input channels or features.
        hidden_channels (list of int): List of hidden layer sizes. The last element is the output size.
        mlp_bias (float): Value for initializing bias terms in linear layers.
        activation_layer (torch.nn.Module, optional): Activation function applied between hidden layers. Default is SiLU.
        bias (bool, optional): If True, the linear layers include bias terms. Default is True.
        dropout (float, optional): Dropout probability applied after the last hidden layer. Default is 0.0 (no dropout).
    '''
    def __init__(self, MLP_configs, bias=True, dropout = 0.0):
        super().__init__()

        in_channels=MLP_configs['in_channels'] 
        hidden_channels=MLP_configs['hidden_channels']
        self.mlp_bias=MLP_configs['mlp_bias']
        activation_layer=MLP_configs['activation_layer']

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if MLP_configs['task'] == 'denoising':
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation_layer())
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout))
        
        self.layers = nn.Sequential(*layers)
        self.layers.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.001)
            torch.nn.init.constant_(m.bias, self.mlp_bias)

    def forward(self, x):
        out = self.layers(x)
        return out
    

class Custom1DFeatureExtractor(nn.Module):
    def __init__(self, im_chans, out_chans):
        super(Custom1DFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=im_chans, out_channels=out_chans[0], kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=out_chans[1], kernel_size=5, stride=1, padding=1, groups=32)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=out_chans[2], kernel_size=7, stride=1, padding=1, groups=64)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1) 

    def forward(self, x):
        x = self.pool1(nn.functional.relu(self.conv1(x)))
        x = self.pool2(nn.functional.relu(self.conv2(x)))
        x = nn.functional.relu(self.conv3(x))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return x
    

class SineLayer(nn.Module):
    '''
    SineLayer is a custom PyTorch module that applies a modified Sinusoidal activation function to the output of a linear transformation
    with adjustable parameters.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If True, the linear transformation includes a bias term. Default is True.
        is_first (bool, optional): If True, initializes the weights with a narrower range. Default is False.
        omega_0 (float, optional): Frequency scaling factor for the sinusoidal activation. Default is 30.
        
    Additional Parameters:
        a_param (float): Exponential scaling factor for the sine function. Controls the amplitude. 
        b_param (float): Exponential scaling factor for the frequency.
        c_param (float): Phase shift parameter for the sine function.
        d_param (float): Bias term added to the output.

    '''
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input, a_param, b_param, c_param, d_param):
        output = self.linear(input)
        output = torch.exp(a_param) * torch.sin(torch.exp(b_param) * self.omega_0 * output + c_param) + d_param
        return output
    

class INR(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features,
                 outermost_linear=True, first_omega_0=30, hidden_omega_0=30, 
                 pos_encode_configs={'type':None, 'use_nyquist': None, 'scale_B': None, 'mapping_input': None},
                 MLP_configs={'model': 'resnet34', 'in_channels': 64, 'hidden_channels': [64, 32, 4], 'activation_layer': nn.SiLU}):
        super().__init__()

        # Positional Encoding
        self.pos_encode = pos_encode_configs['type']
        if self.pos_encode in Encoding().encoding_dict.keys():
            self.positional_encoding = Encoding(self.pos_encode).run(in_features=in_features, pos_encode_configs=pos_encode_configs)
            in_features = self.positional_encoding.out_dim
        elif self.pos_encode == None: 
            self.pos_encode = False
        else:
            assert "Invalid pos_encode. Choose from: [frequency, gaussian]"


        self.ground_truth = MLP_configs['GT']
        self.task = MLP_configs['task']
        self.nonlin = SineLayer
        self.hidden_layers = hidden_layers

        # Harmonizer network
        if MLP_configs['task'] == 'audio':
            self.feature_extractor = torchaudio.transforms.MFCC(
                                                sample_rate=MLP_configs['sample_rate'],
                                                n_mfcc=MLP_configs['in_channels'],
                                                melkwargs={'n_fft': 400, 'hop_length': 160,
                                                            'n_mels': 50, 'center': False})
        elif MLP_configs['task'] == 'shape':
            model_ft = getattr(video, MLP_configs['model'])()
            self.feature_extractor = nn.Sequential(*list(model_ft.children())[:MLP_configs['truncated_layer']])
        elif MLP_configs['task'] == 'inpainting':
            self.feature_extractor = Custom1DFeatureExtractor(im_chans=3, out_chans=[32, 64, 64])
        else:
            model_ft = getattr(models, MLP_configs['model'])()
            self.feature_extractor = nn.Sequential(*list(model_ft.children())[:MLP_configs['truncated_layer']])

        self.aux_mlp = MLP(MLP_configs)

        if MLP_configs['task'] == 'shape':
            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            if MLP_configs['task'] != 'inpainting':
                self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Composer Network
        self.net = []
        self.net.append(self.nonlin(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

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
        
        extracted_features = self.feature_extractor(self.ground_truth)
        if self.task == 'shape':
            gap = self.gap(extracted_features)[:, :, 0, 0, 0]
            coef = self.aux_mlp(gap)
        elif self.task == 'inpainting':
            coef = self.aux_mlp(extracted_features)
        else:
            gap = self.gap(extracted_features.view(extracted_features.size(0), extracted_features.size(1), -1)) 
            coef = self.aux_mlp(gap[..., 0])
        a_param, b_param, c_param, d_param = coef[0]
                
        output = self.net[0](coords, a_param, b_param, c_param, d_param)
        
        for i in range(1, self.hidden_layers + 1):
            output = self.net[i](output, a_param, b_param, c_param, d_param)
        
        output = self.net[self.hidden_layers + 1](output)
                
        return [output, coef]