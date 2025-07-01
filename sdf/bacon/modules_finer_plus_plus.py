import torch
import numpy as np
from torch import nn
import math
from torch.nn import init


def init_weights(m, omega=1, c=1, is_first=False): # Default: Pytorch initialization
    if hasattr(m, 'weight'):
        fan_in = m.weight.size(-1)
        if is_first:
            bound = 1 / fan_in # SIREN
        else:
            bound = math.sqrt(c / fan_in) / omega
        init.uniform_(m.weight, -bound, bound)
        # print('bound:', bound)
    
def init_weights_kaiming(m):
    if hasattr(m, 'weight'):
        init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

def init_bias(m, k):
    if hasattr(m, 'bias'):
        init.uniform_(m.bias, -k, k)

'''Used for SIREN, FINER, Gauss, Wire, etc.'''
def init_weights_cond(init_method, linear, omega=1, c=1, is_first=False):
    init_method = init_method.lower()
    if init_method == 'sine':
        init_weights(linear, omega, 6, is_first)    # SIREN initialization
    elif init_method == 'statistics':
        init_weights(linear, omega, c, is_first)
    elif init_method == 'pytorch_omega':
        init_weights(linear, omega, 1, False)       # multiply 1/omega
    elif init_method == 'pytorch_sine':
        init_weights(linear, omega, 6, False)       # multiply sqrt(6)/omega
    elif init_method == 'pytorch_statistics':
        init_weights(linear, omega, c, False)       # multiply sqrt(c)/omega
    else: ## Default: Pytorch initialization
        pass

def init_bias_cond(linear, fbs=None, hbs=None, is_first=True):
    ''' TODO: bias initialization of hidden layers '''
    if is_first and fbs != None:
        init_bias(linear, fbs)
        # print('fbs:', fbs)
    if not is_first and hbs != None:        
        init_bias(linear, hbs)
    ## Default: Pytorch initialization

''' 
    FINER activation
    TODO: alphaType, alphaReqGrad
'''
def generate_alpha(x, alphaType=None, alphaReqGrad=False):
    with torch.no_grad():
        return torch.abs(x) + 1
    
def finer_activation(x, omega=1, alphaType=None, alphaReqGrad=False):
    return torch.sin(omega * generate_alpha(x, alphaType, alphaReqGrad) * x)

'''
    Gauss activation
'''
def gauss_activation(x, scale):
    return torch.exp(-(scale*x)**2)

def gauss_finer_activation(x, scale, omega, alphaType=None, alphaReqGrad=False):
    return gauss_activation(finer_activation(x, omega, alphaType, alphaReqGrad), scale)

## norm
def gauss_finer_activation_norm(x, scale, omega, alphaType=None, alphaReqGrad=False):
    y = gauss_finer_activation(x, scale, omega, alphaType, alphaReqGrad)
    y_min = gauss_activation(torch.tensor(1), scale)
    y_max = gauss_activation(torch.tensor(0), scale)
    return (y - y_min) / (y_max - y_min)

## previous version
# def rec_gaussian(x, scale):
#     return torch.exp(-(scale*x)**2) - np.exp(-scale**2)

# def rec2_gaussian(x, scale):
#     return rec_gaussian(x, scale) / rec_gaussian(torch.tensor(0.), scale)

# def gauss_finer_activation_norm(x, scale, omega, alphaType=None, alphaReqGrad=False):
#     x = finer_activation(x, omega)
#     return rec2_gaussian(x, scale)
    
'''
    Wire activation
    Wire-Finer ? Default: [Type 2]
        Type 1: cos(omega_w*alpha*x) * exp(-|scale*x|^2)
        Type 2: wire_activation(sin(omega*alpha*x))
'''
def wire_activation(x, scale, omega_w):
    return torch.exp(1j*omega_w*x - torch.abs(scale*x)**2)

def wire_finer_activation(x, scale, omega_w, omega, alphaType=None, alphaReqGrad=False):
    return wire_activation(finer_activation(x, omega, alphaType, alphaReqGrad), scale, omega_w)

def real_wire_activation(x, scale, omega_w):
    return torch.cos(omega_w*x) * torch.exp(-(scale*x)**2)

def real_wire_finer_activation(x, scale, omega_w, omega, alphaType=None, alphaReqGrad=False):
    return real_wire_activation(finer_activation(x, omega, alphaType, alphaReqGrad), scale, omega_w)
'''
    omega
'''
def convert_sigma_scale(x):
    return np.sqrt(0.5) / x

# In order to make the central portions of Gauss and GF activation functions similar.
def omega_centerapproximate(scale, bound=3, yval=0.01):
    _sigma = bound * convert_sigma_scale(scale)
    return np.arcsin(np.sqrt(-np.log(yval + np.exp(-scale**2)))/scale) / _sigma / (_sigma + 1)




## FINER 
class FinerLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, omega=30, 
                 is_first=False, is_last=False, 
                 init_method='sine', init_gain=1, fbs=None, hbs=None, 
                 alphaType=None, alphaReqGrad=False):
        super().__init__()
        self.omega = omega
        self.is_last = is_last ## no activation
        self.alphaType = alphaType
        self.alphaReqGrad = alphaReqGrad
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # init weights
        init_weights_cond(init_method, self.linear, omega, init_gain, is_first)
        # init bias
        init_bias_cond(self.linear, fbs, hbs, is_first)
    
    def forward(self, input):
        wx_b = self.linear(input) 
        if not self.is_last:
            return finer_activation(wx_b, self.omega)
        return wx_b # is_last==True

    def forward_with_interm(self, input):
        wx_b = self.linear(input) 
        if not self.is_last:
            alpha = generate_alpha(wx_b, self.alphaType, self.alphaReqGrad)
            return self.omega*wx_b, self.omega*alpha*wx_b, torch.sin(self.omega*alpha*wx_b)    
        return wx_b # is_last==True
      
class Finer(nn.Module): 
    def __init__(self, in_features=2, out_features=3, hidden_layers=3, hidden_features=256, 
                 first_omega=30, hidden_omega=30, 
                 init_method='sine', init_gain=1, fbs=None, hbs=None, 
                 alphaType=None, alphaReqGrad=False):
        super().__init__()
        self.net = []
        self.net.append(FinerLayer(in_features, hidden_features, is_first=True, 
                                   omega=first_omega, 
                                   init_method=init_method, init_gain=init_gain, fbs=fbs,
                                   alphaType=alphaType, alphaReqGrad=alphaReqGrad))

        for i in range(hidden_layers):
            self.net.append(FinerLayer(hidden_features, hidden_features, 
                                       omega=hidden_omega, 
                                       init_method=init_method, init_gain=init_gain, hbs=hbs,
                                       alphaType=alphaType, alphaReqGrad=alphaReqGrad))

        self.net.append(FinerLayer(hidden_features, out_features, is_last=True, 
                                   omega=hidden_omega, 
                                   init_method=init_method, init_gain=init_gain, hbs=hbs)) # omega: For weight init
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        return self.net(coords)

    def forward_with_interm(self, input):
        interm = {}
        N = len(self.net)
        for idx, layer in enumerate(self.net):
            if idx != N-1:
                wxb, wxb_finer, sin_activated = layer.forward_with_interm(input)
                interm[f'layer_{idx}_wxb'] = wxb
                interm[f'layer_{idx}_wxb_finer'] = wxb_finer
                interm[f'layer_{idx}_sin_acted'] = sin_activated
                out = sin_activated
            else:
                out = layer(input)
                interm[f'layer_{idx}_out'] = out
            input = out
        return interm


## RealWIRE
class RealGaborLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, scale=10, omega_w=20, 
                 is_first=False, is_last=False, 
                 init_method='Pytorch', init_gain=1):
        super().__init__()
        self.scale = scale
        self.omega_w = omega_w
        self.is_last = is_last ## no activation
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # init weights
        init_weights_cond(init_method, self.linear, omega_w, init_gain, is_first)
    
    def forward(self, input):
        wx_b = self.linear(input) 
        if not self.is_last:
            return real_wire_activation(wx_b, self.scale, self.omega_w)
        return wx_b # is_last==True

    def forward_with_interm(self, input):
        wx_b = self.linear(input) 
        if not self.is_last:
            return wx_b, real_wire_activation(wx_b, self.scale, self.omega_w)    
        return wx_b # is_last==True

class RealWire(nn.Module):
    def __init__(self, in_features=2, out_features=3, hidden_layers=3, hidden_features=256, 
                 scale=10, omega_w=20,
                 init_method='sine', init_gain=1):
        super().__init__()
        self.net = []
        self.net.append(RealGaborLayer(in_features, hidden_features, is_first=True, 
                                       scale=scale, omega_w=omega_w, 
                                       init_method=init_method, init_gain=init_gain))

        for i in range(hidden_layers):
            self.net.append(RealGaborLayer(hidden_features, hidden_features, 
                                           scale=scale, omega_w=omega_w, 
                                           init_method=init_method, init_gain=init_gain))

        self.net.append(RealGaborLayer(hidden_features, out_features, is_last=True, 
                                       scale=scale, omega_w=omega_w, 
                                       init_method=init_method, init_gain=init_gain)) # omega_w: For weight init
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        return self.net(coords)

    def forward_with_interm(self, input):
        interm = {}
        N = len(self.net)
        for idx, layer in enumerate(self.net):
            if idx != N-1:
                wxb, wire_activated = layer.forward_with_interm(input)
                interm[f'layer_{idx}_wxb'] = wxb
                interm[f'layer_{idx}_wire_acted'] = wire_activated
                out = wire_activated
            else:
                out = layer(input)
                interm[f'layer_{idx}_out'] = out
            input = out
        return interm


## RealWireFiner
class RealWFLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, scale=10, omega_w=20, omega=1,
                 is_first=False, is_last=False, 
                 init_method='Pytorch', init_gain=1, fbs=None, hbs=None,
                 alphaType=None, alphaReqGrad=False):
        super().__init__()
        self.scale = scale
        self.omega_w = omega_w
        self.omega = omega
        self.is_last = is_last ## no activation
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # init weights
        init_weights_cond(init_method, self.linear, omega*omega_w, init_gain, is_first)
        
        # init bias 
        init_bias_cond(self.linear, fbs, hbs, is_first)
    
    def forward(self, input):
        wx_b = self.linear(input) 
        if not self.is_last:
            return real_wire_finer_activation(wx_b, self.scale, self.omega_w, self.omega)
        return wx_b # is_last==True

    def forward_with_interm(self, input):
        wx_b = self.linear(input) 
        if not self.is_last:
            wxb_finer = self.omega*generate_alpha(wx_b)*wx_b
            sin_acted = torch.sin(wxb_finer)
            wire_acted = real_wire_activation(sin_acted, self.scale, self.omega_w)
            return self.omega*wx_b, wxb_finer, sin_acted, wire_acted
        return wx_b # is_last==True

class RealWF(nn.Module):
    def __init__(self, in_features=2, out_features=3, hidden_layers=3, hidden_features=256, 
                 scale=10, omega_w=20, omega=1,
                 init_method='Pytorch', init_gain=1, fbs=None, hbs=None, 
                 alphaType=None, alphaReqGrad=False):
        super().__init__()
        self.net = []
        self.net.append(RealWFLayer(in_features, hidden_features, is_first=True,
                                    omega=omega, scale=scale, omega_w=omega_w, 
                                    init_method=init_method, init_gain=init_gain, fbs=fbs,
                                    alphaType=alphaType, alphaReqGrad=alphaReqGrad))

        for i in range(hidden_layers):
            self.net.append(RealWFLayer(hidden_features, hidden_features, 
                                        omega=omega, scale=scale, omega_w=omega_w,
                                        init_method=init_method, init_gain=init_gain, hbs=hbs,
                                        alphaType=alphaType, alphaReqGrad=alphaReqGrad))

        self.net.append(RealWFLayer(hidden_features, out_features, is_last=True, 
                                    omega=omega, scale=scale, omega_w=omega_w,
                                    init_method=init_method, init_gain=init_gain, hbs=hbs)) # omega: For weight init
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        return self.net(coords)

    def forward_with_interm(self, input):
        interm = {}
        N = len(self.net)
        for idx, layer in enumerate(self.net):
            if idx != N-1:
                wxb, wxb_finer, sin_acted, wire_acted = layer.forward_with_interm(input)
                interm[f'layer_{idx}_wxb'] = wxb
                interm[f'layer_{idx}_wxb_finer'] = wxb_finer
                interm[f'layer_{idx}_sin_acted'] = sin_acted
                interm[f'layer_{idx}_wire_acted'] = wire_acted
                out = wire_acted
            else:
                out = layer(input)
                interm[f'layer_{idx}_out'] = out
            input = out
        return interm  



## Gauss
class GaussLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, scale=30.0,
                 is_first=False, is_last=False,
                 init_method='Pytorch', init_gain=1):
        super().__init__()
        self.scale = scale
        self.is_last = is_last
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # init weights
        init_weights_cond(init_method, self.linear, None, init_gain, is_first)
    
    def forward(self, input):
        wx_b = self.linear(input) 
        if not self.is_last:
            return gauss_activation(wx_b, self.scale)
        return wx_b # is_last==True
    
    def forward_with_interm(self, input):
        wx_b = self.linear(input) 
        if not self.is_last:
            return wx_b, gauss_activation(wx_b, self.scale)
        return wx_b # is_last==True
    
class Gauss(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, 
                 scale=30,
                 init_method='Pytorch', init_gain=1,
                 is_sdf=True):
        super().__init__()
        self.is_sdf = is_sdf
        self.net = []
        self.net.append(GaussLayer(in_features, hidden_features, is_first=True, 
                                   scale=scale,
                                   init_method=init_method, init_gain=init_gain))

        for i in range(hidden_layers):
            self.net.append(GaussLayer(hidden_features, hidden_features, 
                                       scale=scale,
                                       init_method=init_method, init_gain=init_gain))
            
        self.net.append(GaussLayer(hidden_features, out_features, is_last=True, 
                                   scale=scale,
                                   init_method=init_method, init_gain=init_gain))
        self.net = nn.Sequential(*self.net)
    
    def forward(self, model_input):
        # output = self.net(coords)
        # return output
        coords = model_input['coords']
        output = self.net(coords)
        
        if self.is_sdf:
            return {'model_in': model_input, 'model_out': output}
        else:
            return {'model_in': model_input, 'model_out': {'output': output}}      


## GFINER: Gauss-Finer
class GFLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, scale=3, omega=1,
                 is_first=False, is_last=False, 
                 init_method='Pytorch', init_gain=1, fbs=None, hbs=None,
                 alphaType=None, alphaReqGrad=False):
        super().__init__()
        self.scale = scale
        self.omega = omega
        self.is_last = is_last
        self.linear = nn.Linear(in_features, out_features, bias=bias)
            
        # init weights
        init_weights_cond(init_method, self.linear, omega, init_gain, is_first)
        # if not is_last:
            # init_weights_cond(init_method, self.linear, omega, init_gain, is_first)
        # else:
            # init_weights_cond(init_method, self.linear, 30, init_gain, is_first)
            
        # init bias 
        init_bias_cond(self.linear, fbs, hbs, is_first)
    
    def forward(self, input):
        wx_b = self.linear(input) 
        if not self.is_last:
            return gauss_finer_activation_norm(wx_b, self.scale, self.omega)
        return wx_b # is_last==True

    def forward_with_interm(self, input):
        wx_b = self.linear(input) 
        if not self.is_last:
            wxb_finer = self.omega*generate_alpha(wx_b)*wx_b
            sin_acted = torch.sin(wxb_finer)
            gas_acted = gauss_finer_activation_norm(wx_b, self.scale, self.omega)
            return self.omega*wx_b, wxb_finer, sin_acted, gas_acted
        return wx_b # is_last==True

class GF(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers, hidden_features, 
                 scale=3, omega=1, 
                 init_method='Pytorch', init_gain=1, fbs=None, hbs=None, 
                 alphaType=None, alphaReqGrad=False, is_sdf=True):
        super().__init__()
        self.is_sdf = is_sdf
        self.net = []
        self.net.append(GFLayer(in_features, hidden_features, is_first=True, 
                                scale=scale, omega=omega, 
                                init_method=init_method, init_gain=init_gain, fbs=fbs,
                                alphaType=alphaType, alphaReqGrad=alphaReqGrad))
        
        for i in range(hidden_layers):
            self.net.append(GFLayer(hidden_features, hidden_features, 
                                     scale=scale, omega=omega, 
                                     init_method=init_method, init_gain=init_gain, hbs=hbs,
                                     alphaType=alphaType, alphaReqGrad=alphaReqGrad))
         
        self.net.append(GFLayer(hidden_features, out_features, is_last=True, 
                                scale=scale, omega=omega, 
                                init_method=init_method, init_gain=init_gain, hbs=hbs)) # omega: For weight init
        self.net = nn.Sequential(*self.net)
    
    def forward(self, model_input):
        coords = model_input['coords']
        output = self.net(coords)
        
        if self.is_sdf:
            return {'model_in': model_input, 'model_out': output}
        else:
            return {'model_in': model_input, 'model_out': {'output': output}}        
    



## WFINER
class WFLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, scale=10, omega_w=20, omega=1,
                 is_first=False, is_last=False, 
                 init_method='Pytorch', init_gain=1, fbs=None, hbs=None,
                 alphaType=None, alphaReqGrad=False):
        super().__init__()
        self.scale = scale
        self.omega_w = omega_w
        self.omega = omega
        self.is_last = is_last ## no activation
        dtype = torch.float if is_first else torch.cfloat
        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
        
        # init weights
        init_weights_cond(init_method, self.linear, omega*omega_w, init_gain, is_first)
        
        # init bias 
        init_bias_cond(self.linear, fbs, hbs, is_first)
        
    def forward(self, input):
        wx_b = self.linear(input) 
        if not self.is_last:
            return wire_finer_activation(wx_b, self.scale, self.omega_w, self.omega)
        return wx_b # is_last==True


class WF(nn.Module):
    def __init__(self, in_features=2, out_features=3, hidden_layers=3, hidden_features=256, 
                 scale=10, omega_w=20, omega=1,
                 init_method='Pytorch', init_gain=1, fbs=None, hbs=None, is_sdf=True,
                 alphaType=None, alphaReqGrad=False):
        super().__init__()
        hidden_features = int(hidden_features / np.sqrt(2))
        self.is_sdf = is_sdf
        self.net = []
        self.net.append(WFLayer(in_features, hidden_features, is_first=True,
                                    omega=omega, scale=scale, omega_w=omega_w, 
                                    init_method=init_method, init_gain=init_gain, fbs=fbs,
                                    alphaType=alphaType, alphaReqGrad=alphaReqGrad))

        for i in range(hidden_layers):
            self.net.append(WFLayer(hidden_features, hidden_features, 
                                        omega=omega, scale=scale, omega_w=omega_w,
                                        init_method=init_method, init_gain=init_gain, hbs=hbs,
                                        alphaType=alphaType, alphaReqGrad=alphaReqGrad))

        self.net.append(WFLayer(hidden_features, out_features, is_last=True, 
                                    omega=omega, scale=scale, omega_w=omega_w,
                                    init_method=init_method, init_gain=init_gain, hbs=hbs)) # omega: For weight init
        self.net = nn.Sequential(*self.net)

    def forward(self, model_input):
        coords = model_input['coords']
        output = self.net(coords)
        output = output.real
        
        if self.is_sdf:
            return {'model_in': model_input, 'model_out': output}
        else:
            return {'model_in': model_input, 'model_out': {'output': output}}        
    