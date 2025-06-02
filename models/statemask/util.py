from numpy.random import uniform
import torch
import numpy as np
import torch.nn as nn
from collections import Counter

def mask_select(data, mask_op, data_type='torch'):
    '''
    data: numpy array (sample, dims, steps)\n
    im_op: list lens== steps
    '''
    
    Pos_num = Counter(mask_op)[1]
    if Pos_num ==  0:
        mask_op[-1] = 1
    
    _data = []
    for id, tag in enumerate(mask_op):
        if int(tag) == 1:
            _data.append(data[:,:,id])
    if data_type == 'numpy':
        select_data = np.stack(_data, axis=-1)
    elif data_type == 'torch':
        select_data = torch.stack(_data, dim=2)
    else:
        raise ValueError("Unsupported data_type: {}".format(type(data)))
    
    return select_data

def sigmoid_act(v):
    sv = 1 / (1+np.exp(-v))
    x = 1.0 if uniform(0, 1) > sv else 0.0
    return x

def state2config(state):
    config = {}
    dim = state.shape[0]
    for i, d in enumerate(list(range(dim))):
        config['stateMask_{}'.format(i)] = int(state[d])
    
    return config


class Praticle2fitness:
    def setup(self, data = None) -> None:
        self.train_data = data.train_data
        self.valid_data = data.valid_data
        
        # self.hidden_states = data.hidden_states
        # self.target_seqs = data.target_seqs
        self.Reg_lambda = data.Reg_lambda if 'Reg_lambda' in data.dict else 0
        self.loss = nn.MSELoss()
        
        self.problem_dim = self.train_data.hidden_states.size(2)
        # self.stateMask = [config['stateMask_{}'.format(i)] for i in range(self.hidden_states.size(2))]
        # self.hiddens = self.state_select(self.train_data.hidden_states, self.stateMask)
        # self.tgts = self.state_select(self.train_data.target_seqs, self.stateMask)
        
    def step(self,):
        # hiddens = self.state_select(self.train_data.hidden_states, self.stateMask)
        # tgts = self.state_select(self.train_data.target_seqs, self.stateMask)
        
        readout, tra_mse = self.cal_fitness(h=self.hiddens, y=self.tgts)
        
        val_hidden = self.valid_data.hidden_states[:,:, -1]
        val_y = self.valid_data.target_seqs[:,:, -1]
        
        val_pred =readout(val_hidden)
        val_mse = self.loss(val_pred, val_y).item()
        
        return {
            'tmse': round(np.sqrt(tra_mse),6),
            'vmse': round(np.sqrt(val_mse),6),
        }
    
    @staticmethod
    def state_select(state, stateMask):
        assert len(stateMask) == state.shape[2]
        selection = mask_select(state, stateMask, data_type='torch')
        selection = selection.permute(0,2,1)
        selection = torch.flatten(selection, start_dim=0, end_dim=1)
        return selection  
    
    @staticmethod
    def cal_fitness(h, y, Reg_lambda=0):
        loss = nn.MSELoss()
        input_size = h.size(1)
        tgt_size = y.size(1)
        
            # def solve_output(self, Hidden_States, y):
        t_hs = torch.cat((torch.ones(h.size(0), 1), h), dim=1)
        HTH = t_hs.T @ t_hs
        HTY = t_hs.T @ y
        # ridge regression with pytorch
        I = (Reg_lambda * torch.eye(HTH.size(0)))
        A = HTH + I
        # orig_rank = torch.matrix_rank(HTH).item() # torch.matrix_rank is deprecated in favor of torch.linalg.matrix_rank in v1.9.0
        orig_rank = torch.linalg.matrix_rank(A, hermitian=True).item()
        tag = 'Inverse' if orig_rank == t_hs.size(1) else 'Pseudo-inverse'
        if tag == 'Inverse':
            W = torch.mm(torch.inverse(A), HTY).t()
        else:
            try:
                # W = torch.mm(torch.linalg.pinv(A.to(self.device)),
                #             HTY.to(self.device)).t()
                W = torch.linalg.lstsq(A.cpu(), HTY.cpu(), driver = 'gelsd').solution.T
            except:
                W = torch.mm(torch.linalg.pinv(A.cpu()),
                            HTY.cpu()).t()
        
        readout = nn.Linear(in_features=input_size, out_features=tgt_size)
        readout.bias = nn.Parameter(W[:, 0], requires_grad=False)
        readout.weight = nn.Parameter(W[:, 1:], requires_grad=False)
        readout.eval()
        pred = readout(h)
        fitness= loss(pred, y).item()
        
        return readout, fitness
    
    def reset_config(self, new_config):        
        self.stateMask = [new_config['stateMask_{}'.format(i)] for i in range(self.problem_dim )]
        self.hiddens = self.state_select(self.train_data.hidden_states, self.stateMask)
        self.tgts = self.state_select(self.train_data.target_seqs, self.stateMask)
        
        return True
        