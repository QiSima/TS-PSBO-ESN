import os
import sys

from numpy.lib.function_base import select
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import gc
# from tqdm import trange
import torch
import torch.nn as nn
import numpy as np
import scipy 

class fcLayer(nn.Module):
    def __init__(self, in_features, out_features, device, bias=True):
        super(fcLayer, self).__init__()
        self.fc = nn.Linear(in_features,out_features, bias)
        self.device = device
        self.fc.to(self.device)
        self.bias = bias
        
    
    def freeze(self, ):
        self.fc.weight.requires_grad = False
        if self.bias:
            self.fc.bias.requires_grad = False
        
    def active(self,):
        self.fc.weight.requires_grad = True
        if self.bias:
            self.fc.bias.requires_grad = True
    
    def update(self, weight, bias = None):
        '''
        Attention! If the parameters are updated with non-gradient, the freezed() need be called after the update(). like that:\n
        ---
        ho.update(weight, bias)\n
        ho.freeze()\n
        ---
        '''
        self.fc.weight = nn.Parameter(weight.to(self.device))
        if self.bias:
            self.fc.bias = nn.Parameter(bias.to(self.device))
        
    def forward(self, x):
        fm = self.fc(x)
        return fm

class ESNCell(nn.Module):
    """Echo state network Cell 

    Args:
        init: The method of initializing hidden weight, supporting: 'vanilla', 'svd'.
        hidden_size: The number of features in the hidden state h.
        input_dim: The number of expected features in the input x of each time step.
        nonlinearity: The non-linearity to use ['tanh'|'relu'|'sigmoid'].
        leaky_r: Leaking rate of reservoir's neurons.
        weight_scale: Desired spectral radius of recurrent weight matrix.
        iw_bound: The bound (tuple) of the uniform distribution of input weight.
        hw_bound: The bound (tuple) of the uniform distribution of hidden weight.
        device: The device of computing the variables.

    Inputs:
        current_input (batch, input_dim): tensor containing the features of
            the input at t time step.
        last_state (batch, hidden_size): tensor containing
                the initial reservoir's hidden state.

    Outputs: 
        _last_state (batch, hidden_size): tensor containing the hidden state at t time step .
    """
    def __init__(self, init = 'vanilla', hidden_size = 500, input_dim = 1, nonlinearity = 'relu', weight_scale = 0.9, iw_bound = (-0.1, 0.1), hw_bound=(-1,1), sparsity = 1, device='cpu'):
        super(ESNCell, self).__init__()
        
        self.init_method = init
        self.Hidden_Size = hidden_size
        self.input_dim = input_dim
        self.device = device
        self.Weight_scaling = weight_scale
        self.iw_bound = iw_bound
        self.hw_bound = hw_bound
        self.nonlinearity = nonlinearity
        self.sparsity = sparsity
        
        self.ih = nn.Linear(self.input_dim, self.Hidden_Size, bias=False).to(self.device)
        self.hh = nn.Linear(
                self.Hidden_Size, self.Hidden_Size, bias=False).to(self.device)
        
        if self.nonlinearity == 'tanh':
            self.act_f = torch.tanh
        elif self.nonlinearity == 'relu':
            self.act_f = torch.relu
        elif self.nonlinearity == 'sigmoid':
            self.act_f = torch.sigmoid
        else:
            raise ValueError(
                "Unknown nonlinearity '{}'".format(nonlinearity))
        
        self.init_weights()

        
    def init_weights(self, ):
        w_ih = torch.empty(self.Hidden_Size, self.input_dim).to(self.device)
        if isinstance(self.iw_bound, tuple):
            nn.init.uniform_(w_ih,self.iw_bound[0],self.iw_bound[1])
        elif isinstance(self.iw_bound, float):
            nn.init.uniform_(w_ih,-self.iw_bound,self.iw_bound)
        else:
            raise ValueError('Invalid iw_bound: {}'.format(self.iw_bound))
        
        if self.init_method == 'svd':
            w_hh = self.svd_init()
        elif self.init_method == 'vanilla':
            w_hh = self.vanilla_init()
        elif self.init_method == 'sparse':
            w_hh = self.sparse_init()
        else:
            raise ValueError(
                "Unknown hidden init '{}'".format(self.init_method))
            
        self.ih.weight = nn.Parameter(w_ih)
        self.hh.weight = nn.Parameter(w_hh)
        self.freeze()
        
    def svd_init(self, singular_values = None):
        svd_u = torch.empty(
            self.Hidden_Size, self.Hidden_Size).to(self.device)
        svd_v = torch.empty(
            self.Hidden_Size, self.Hidden_Size).to(self.device)
        # 填充正交矩阵，非零元素依据均值0，标准差std的正态分布生成
        nn.init.orthogonal_(svd_u)
        nn.init.orthogonal_(svd_v)

        #生成对角化矩阵
        if singular_values is None:
            _svd_s = torch.empty(self.Hidden_Size).to(self.device)
            if isinstance(self.hw_bound, tuple):
                nn.init.uniform_(_svd_s,self.hw_bound[0],self.hw_bound[1])
            elif isinstance(self.hw_bound, float):
                nn.init.uniform_(_svd_s,-self.hw_bound,self.hw_bound)
            else:
                raise ValueError('Invalid hw_bound: {}'.format(self.hw_bound))
        else:
            _svd_s = singular_values.detach().clone().to(self.device)
            assert _svd_s.size(0) == self.Hidden_Size
        

        _svd_s = torch.diag(_svd_s).to(self.device)
        assert len(_svd_s.size()) == 2

        Hidden_weight = svd_u.mm(_svd_s).mm(svd_v)
        Hidden_weight = self.Weight_scaling * Hidden_weight
        # Hidden_weight = Hidden_weight.to(self.device)
        
        return Hidden_weight
    
    def vanilla_init(self,):
        w_hh = torch.empty(self.Hidden_Size, self.Hidden_Size).to(self.device)

        if isinstance(self.hw_bound, tuple):
            nn.init.uniform_(w_hh,self.hw_bound[0],self.hw_bound[1])
        elif isinstance(self.hw_bound, float):
            nn.init.uniform_(w_hh,-self.hw_bound,self.hw_bound)
        else:
            raise ValueError('Invalid hw_bound: {}'.format(self.hw_bound))
              
        spectral_radius = torch.max(torch.abs(torch.linalg.eigvals(w_hh).real))
        w_hh = w_hh * self.Weight_scaling / spectral_radius
        return w_hh
    
    def sparse_init(self,):
        w_hh_temp = scipy.sparse.rand(self.Hidden_Size, self.Hidden_Size, density = 1 - self.sparsity, format='csc').toarray()
        w_hh_temp[np.where(w_hh_temp != 0)] -= 0.5
        
        w_hh_temp = torch.tensor(w_hh_temp).to(torch.float32).to(self.device)
        spectral_radius = torch.max(torch.abs(torch.linalg.eigvals(w_hh_temp).real))
        w_hh = w_hh_temp * self.Weight_scaling / spectral_radius
        return w_hh
        
    def forward(self, current_input, last_state):
        current_state = self.ih(current_input) + self.hh(last_state)
        current_state = self.act_f(current_state)
        return current_state

    def freeze(self, ):
        self.ih.weight.requires_grad = False
        self.hh.weight.requires_grad = False     
        
    def active(self,):
        self.ih.weight.requires_grad = True
        self.hh.weight.requires_grad = True     



class esnLayer(nn.Module):
    '''input: \tx, shape: (N_samples, dimensions, steps ) \n      
    return: 
        \tlayer_hs, shape: (N_samples, hidden_size, steps)\n
        \tlast_hs, shape: (N_samples, hidden_size)        
    '''
    def __init__(self, init = 'vanilla', hidden_size = 500, input_dim = 1, nonlinearity = 'sigmoid', leaky_r = 1, weight_scale = 0.9, iw_bound = (-0.1, 0.1), hw_bound=(-1,1), sparsity = 1, device='cpu'):
        super(esnLayer, self).__init__()
        self.esnCell = ESNCell(
            init=init,
            hidden_size=hidden_size,
            input_dim=input_dim,
            nonlinearity=nonlinearity,
            weight_scale=weight_scale,
            iw_bound=iw_bound,
            hw_bound=hw_bound,
            device=device,
            sparsity = sparsity
        )
        self.leaky_r = leaky_r
        self.Hidden_Size = hidden_size
        self.device = device
    
    def init_weights(self,):
        self.esnCell.init_weights()
    
    def get_weights(self,):
        '''For debug'''
        return [self.esnCell.ih.weight, self.esnCell.hh.weight]
    
    
    def forward(self, x, _last_state = None):
        '''
        return: layer_hidden_state, last_state\n
        layer_hidden_state: (samples,Hidden_Size, time_steps)\n
        _last_state: (samples, Hidden_Size )
        '''
        # with torch.no_grad():
        samples, time_steps = x.shape[0], x.shape[2]
        layer_hidden_state = torch.empty(samples,self.Hidden_Size, time_steps).to(self.device)
        
        # layer_hidden_state = []
        if _last_state is None:
            last_state = torch.zeros(samples, self.Hidden_Size).to(self.device)
        else:
            assert _last_state.shape[1] == self.Hidden_Size
            last_state = _last_state.detach().clone().to(self.device)
        
        for t in range(time_steps):
            current_state = self.esnCell(x[:,:,t],  last_state)
            last_state = (1- self.leaky_r) * last_state + self.leaky_r * current_state
            layer_hidden_state[:,:,t] = last_state
            torch.cuda.empty_cache() if next(self.parameters()).is_cuda else gc.collect()
        
        
        # layer_hidden_state = torch.stack(layer_hidden_state, dim=2)
        torch.cuda.empty_cache() if next(self.parameters()).is_cuda else gc.collect()
        
        return layer_hidden_state, last_state
    
    def freeze(self, ):
        self.esnCell.freeze()
        
    def active(self,):
        self.esnCell.active()