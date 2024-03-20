import torch
from torch import nn
import numpy as np

## FINER SIREN WIRE GUASS PEMLP


## SIREN
class SineLayer(nn.Module):
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

    def forward(self, input):
        out = torch.sin(self.omega_0 * self.linear(input))
        return out

class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, first_omega_0=30, hidden_omega_0=30.0, bias=True):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0, bias=bias))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                          np.sqrt(6 / hidden_features) / hidden_omega_0)
        self.net.append(final_linear)

        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        return output
    
    
## FINER sin⁡(𝑜𝑚𝑒𝑔𝑎∗𝑠𝑐𝑎𝑙𝑒∗(𝑊𝑥+𝑏𝑖𝑎𝑠)) 𝑠𝑐𝑎𝑙𝑒=|𝑊𝑥+𝑏𝑖𝑎𝑠|+1
class FinerLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30, first_bias_scale=None, scale_req_grad=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
        self.scale_req_grad = scale_req_grad
        self.first_bias_scale = first_bias_scale
        if self.first_bias_scale != None:
            self.init_first_bias()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def init_first_bias(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.bias.uniform_(-self.first_bias_scale, self.first_bias_scale)
                # print('init fbs', self.first_bias_scale)

    def generate_scale(self, x):
        if self.scale_req_grad: 
            scale = torch.abs(x) + 1
        else:
            with torch.no_grad():
                scale = torch.abs(x) + 1
        return scale
        
    def forward(self, input):
        x = self.linear(input)
        scale = self.generate_scale(x)
        out = torch.sin(self.omega_0 * scale * x)
        return out
    

class Finer(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, first_omega_0=30, hidden_omega_0=30.0, bias=True, 
                 first_bias_scale=None, scale_req_grad=False):
        super().__init__()
        self.net = []
        self.net.append(FinerLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0, first_bias_scale=first_bias_scale, scale_req_grad=scale_req_grad))

        for i in range(hidden_layers):
            self.net.append(FinerLayer(hidden_features, hidden_features, omega_0=hidden_omega_0, scale_req_grad=scale_req_grad))

        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                          np.sqrt(6 / hidden_features) / hidden_omega_0)
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output        
       
 
## Gauss
class GaussLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, scale=30.0):
        super().__init__()
        self.in_features = in_features
        self.scale = scale
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input):
        return torch.exp(-(self.scale * self.linear(input))**2)
    
class Gauss(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, scale=30.0):
        super().__init__()
        self.nonlin = GaussLayer
            
        self.net = []
        self.net.append(self.nonlin(in_features, hidden_features, scale=scale))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features, scale=scale))

        final_linear = nn.Linear(hidden_features, out_features)                
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        return output

        

## WIRE Ref：https://github.com/vishwa91/wire
class ComplexGaborLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega0=10.0, sigma0=40.0, trainable=False):
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
        
        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
    
    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        return torch.exp(1j*omega - scale.abs().square())
    
class Wire(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, hidden_layers, first_omega_0=20, hidden_omega_0=20, scale=10.0):
        super().__init__()
        # All results in the paper were with the default complex 'gabor' nonlinearity
        self.nonlin = ComplexGaborLayer
        
        # Since complex numbers are two real numbers, reduce the number of hidden parameters by 2
        hidden_features = int(hidden_features / np.sqrt(2))
        dtype = torch.cfloat
        self.complex = True
        
        self.net = []
        self.net.append(self.nonlin(in_features, hidden_features, is_first=True, omega0=first_omega_0, sigma0=scale))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features, omega0=hidden_omega_0, sigma0=scale))

        final_linear = nn.Linear(hidden_features, out_features, dtype=dtype)            
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        return output.real
    

## PEMLP
class PositionalEncoding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        super().__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs) 
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]
        return torch.cat(out, -1)
    

class PEMLP(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, N_freqs=10):
        super().__init__()
        self.enconding = PositionalEncoding(in_channels=in_features, N_freqs=N_freqs)
        
        self.net = []
        self.net.append(nn.Linear(self.enconding.out_channels, hidden_features))
        self.net.append(nn.ReLU(True))

        for i in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.ReLU(True))

        final_linear = nn.Linear(hidden_features, out_features)                
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(self.enconding(coords))
        return output

    

    

    
    
    