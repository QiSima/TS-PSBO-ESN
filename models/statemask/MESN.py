import os,sys,gc
from models.Base import esnLayer
from models.stochastic.ESN import EchoStateNetwork
import numpy as np
from collections import Counter
from  task.TaskLoader import Opt, rnn_dataset, torch_dataloader
import torch
import torch.nn as nn
from task.metric import rmse
from task.util import set_logger
import copy

def mask_select(data, mask_op, data_type='numpy'):
    '''
    data: numpy array (sample, dims, steps)\n
    im_op: list lens== steps
    '''
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


class MaskESN(EchoStateNetwork):
    def __init__(self, opts=None, logger=None):
        super().__init__(opts, logger)
        self.init_inputMask()
        self.init_stateMask()
        self.init_hiddenMask()

    def init_hiddenMask(self, maskcode = None):
        hidden_select = False
        if maskcode is None: 
            for i in range(self.readout_size):
                if 'stripH_{}'.format(i) in self.opts.dict:
                    hidden_select = True
                    break
            
            if hidden_select:
                for i in range(self.readout_size):
                    if 'stripH_{}'.format(i) not in self.opts.dict:
                        raise ValueError('Missing hyper config: "stripH_{}"!'.format(i))
                self.stripH_operator = [self.opts.dict['stripH_{}'.format(i)] for i in range(self.readout_size)]
            else:
                self.stripH_operator = np.ones(self.readout_size).tolist()
        else:
            self.stripH_operator = copy.deepcopy(maskcode)

        for i, o in enumerate(self.stripH_operator):
            if o not in [0,1]:
                raise ValueError('Non-supported hyper config: "stripH_{}" with {}'.format(i, o))
        
        hiddenSelected_size = self.stripH_operator.count(1)
        
        self.readout_size = hiddenSelected_size

        self.readout = nn.Linear(
            self.readout_size, self.Output_dim)
        self.readout.weight.requires_grad = False
        self.readout.bias.requires_grad = False

    def init_inputMask(self):
        input_select_tag = False
        for i in range(self.Time_steps):
            if 'inputMask_{}'.format(i) in self.opts.dict:
                input_select_tag = True
                break
        if input_select_tag:
            for i in range(self.Time_steps):
                if 'inputMask_{}'.format(i) not in self.opts.dict:
                    raise ValueError('Missing hyper config: "inputMask_{}"!'.format(i))
            
            self.inputMask = [self.opts.dict['inputMask_{}'.format(i)] for i in range(self.Time_steps)]
        else:
            inputMask = np.ones(self.Time_steps)
            self.inputMask = inputMask.tolist()
        
        inputStep_count = Counter(self.inputMask)
        self.inputMask_Pos = inputStep_count[1]
        
        if self.inputMask_Pos < 2:
            if self.inputMask[-1] == 1:
                self.inputMask[-2] = 1
            else:
                self.inputMask[-1] = 1
        
        if self.inputMask_Pos - self.Readout_steps < 0:
            neg_ids = [i for i,v in enumerate(self.inputMask) if v == 0]
            fill_num = (self.inputMask_Pos - self.Readout_steps ) * -1
            
            rev_ids = np.random.choice(neg_ids, size = fill_num, replace = False).tolist()
            for id in rev_ids:
                self.inputMask[id] = 1
            
        inputStep_count = Counter(self.inputMask)
        self.inputMask_Pos = inputStep_count[1]
        
        assert self.inputMask_Pos - self.Readout_steps >=0   
        
    def init_stateMask(self, maskcode = None):
        state_select_tag = False
        if maskcode is None:
            for i in range(self.inputMask_Pos):
                if 'stateMask_{}'.format(i) in self.opts.dict:
                    state_select_tag = True
                    break
            if state_select_tag:
                for i in range(self.inputMask_Pos):
                    if 'stateMask_{}'.format(i) not in self.opts.dict:
                        raise ValueError('Missing hyper config: "stateMask_{}"!'.format(i))

                self.stateMask = [self.opts.dict['stateMask_{}'.format(i)] for i in range(self.inputMask_Pos)]
            else:
                stateMask = np.zeros(self.inputMask_Pos)
                stateMask[-self.Readout_steps:] = 1
                self.stateMask = stateMask.tolist()
        else:
            self.stateMask = copy.deepcopy(maskcode)
        
        assert len(self.stateMask) == self.inputMask_Pos
        
        stateStep_count = Counter(self.stateMask)
        self.stateMask_Pos = stateStep_count[1]
        if self.stateMask_Pos == 0:
            self.stateMask[-1] = 1
            stateStep_count = Counter(self.stateMask)
            self.stateMask_Pos = stateStep_count[1]
        
    def data_loader(self, data, _batch_size = None):
        '''
        Transform the numpy array data into the pytorch data_loader
        '''
        data_batch_size = self.opts.batch_size if _batch_size is None else _batch_size
        set_data = rnn_dataset(data, self.Output_dim, self.Lag_order,self.Input_dim)
        set_if_x_data = mask_select(set_data.data,self.inputMask)
        set_if_y_data = mask_select(set_data.label, self.inputMask)
        
        set_data.data = set_if_x_data
        set_data.label = set_if_y_data
        
        set_loader = torch_dataloader(set_data, batch_size= data_batch_size,cuda= self.usingCUDA)
        return set_loader        

    def state_select(self, state):
        assert len(self.stateMask) == state.shape[2]
        selection = mask_select(state, self.stateMask, data_type='torch')
        selection = selection.permute(0,2,1)
        selection = torch.flatten(selection, start_dim=0, end_dim=1).to(self._device)
        return selection

    def stripH_process(self, hidden):
        '''data shape: samples, dim, steps
        '''          
        select = []
        read_operator = self.stripH_operator.copy()
        
        assert hidden.shape[1] == len(read_operator)
        for i, tag in enumerate(read_operator):
            if int(tag) == 1:
                select.append(hidden[:, i, :])
        
        if len(select) == 0:
            select = hidden
        else:
            select = torch.stack(select, dim=1).to(self._device)
        
        return select
        
    def io_check(self, hidden, x):
        x = x.to(self._device)
        if self.fc_io == 'step':
            hidden = torch.cat(
                (x, hidden), dim=1)
        elif self.fc_io == 'series':
            series_x = self.gen_series_x(x, self._device)
            full_x = series_x.unsqueeze(2).repeat(1,1, hidden.shape[2])
            hidden = torch.cat((full_x, hidden), dim=1)
        
        hidden = self.stripH_process(hidden)
        
        return hidden 
        
    def update_readout(self, f_Hidden, x, y):
        f_Hidden = self.io_check(f_Hidden, x)
        f_Hidden = self.state_select(f_Hidden)
        y = self.state_select(y)
        
        W, tag = self.solve_output(f_Hidden, y)
        self.readout.bias = nn.Parameter(W[:, 0], requires_grad=False)
        self.readout.weight = nn.Parameter(W[:, 1:], requires_grad=False)
        self.logger.info('Global LSM: {} \t L2 regular: {}'.format(
            tag, 'True' if self.opts.reg_lambda != 0 else 'False'))
            

    def xfit(self, train_data, val_data):
        train_loader = self.data_loader(train_data)
        val_loader = self.data_loader(val_data)
        h_states, x, y = self.batch_transform(train_loader)
        torch.cuda.empty_cache() if next(self.parameters()).is_cuda else gc.collect()
        self.update_readout(h_states, x, y)
        pred = self.readout(self.io_check(h_states, x)[:,:,-1])
        y = y[:,:,-1].cpu().numpy()
        pred = pred.cpu().numpy()
        self.fit_info.trmse = rmse(y,pred)
        self.eval()
        _, val_y, vpred = self.loader_pred(val_loader)

        self.fit_info.vrmse = rmse(val_y, vpred)

        self.xfit_logger()

        torch.cuda.empty_cache() if next(self.parameters()).is_cuda else gc.collect()

        return self.fit_info