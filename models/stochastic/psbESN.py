import os
import sys
from typing import Any
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from task.TaskLoader import Opt


import torch
import numpy as np

import copy
from ray import tune
from models.statemask.MESN import MaskESN
from task.util import os_makedirs,os_rmdirs
import pandas as pd

from ray import tune

from models.statemask.pop import PSB
from models.statemask.bayes import bayesTuner


class PSB_ESN(Opt):
    def __init__(self, opts, logger):
        super().__init__()
        self.opts = opts
        self.logger = logger
        
        self.host_dir = self.opts.model_fit_dir
        self.model_name = self.opts.model_name
        
        for (arg, value) in opts.dict.items():
            self.logger.info("Argument %s: %r", arg, value)
            
        self.model_opts = opts.esn_opts
        self.model_opts.hyper.H = self.opts.H
        self.model_opts.hyper.lag_order = self.opts.lag_order
        self.model_opts.hyper.batch_size = self.opts.batch_size
        self.model_opts.hyper.device = self.opts.device
        
        self.fit_info = Opt()
        
        self.pop_size = opts.pop_size if 'pop_size' in opts.dict else 100
        self.maxCycle = opts.maxCycle if 'maxCycle' in opts.dict else 100
        self.bayes_tag = opts.bayes if 'bayes' in opts.dict else True
        
        self.problem_dims = self.opts.lag_order
        
        self.tuning_dict = {'stateMask_{}'.format(i) :  tune.choice([0,1]) for i in range(self.problem_dims)}
        
        self.bs_epochs = opts.bs_epochs if 'bs_epochs' in opts.dict else 10
        
        self.psb_init = opts.psb_init if 'psb_init' in opts.dict else ''
        self.psb_init.replace('series0', f'series{opts.sid}')
        
    def optimize(self, train_data, valid_data):
        Maskcode = []
        psb = PSB(pop_size = self.pop_size, maxCycle = self.maxCycle, bayes = self.bayes_tag, bs_epochs=self.bs_epochs, train_data = train_data, valid_data = valid_data, save_dir = self.tuner_dir, tuning_dict=self.tuning_dict, init_file=self.psb_init, logger = self.logger)
        
        psb.evolve()
        
        
        history = psb.history.fitness_history
        # history_file = os.path.join(self.tuner_dir, 'fitness_history.npz')
        # if not os.path.exists(history_file):
        #     np.savez(history_file, fitness = history, gbest = gbest_history)
        
        nums = history.shape[1]
        c_name = ['pop_{}'.format(i) for i in range(nums)]
        df = pd.DataFrame(history, columns=c_name)
        df.to_csv(os.path.join(self.ckp_dir, 'fitness_history.csv'))
        
        pbest_history =[]
        for i in range(self.pop_size):
            pbest_fitness = psb.history.pbest[i].history.reshape(-1,1)
            pbest_history.append(pbest_fitness)
        pbest_history = np.concatenate(pbest_history, axis = 1)
        gbest_history = psb.history.gbest.history
        best_history = np.concatenate((gbest_history.reshape(-1,1), pbest_history), axis = 1)
        c_name.insert(0, 'gbest')
        df = pd.DataFrame(best_history, columns=c_name)
        df.to_csv(os.path.join(self.ckp_dir, 'pbest_history.csv'))
        
        c_name = ['stateMask_{}'.format(i) for i in range(self.problem_dims)]
        c_name.insert(0, 'gbest')
        state_history = psb.history.gbest.state_history
        best_history = np.concatenate((gbest_history.reshape(-1,1), state_history),axis = 1)
        df = pd.DataFrame(best_history, columns=c_name)
        df.to_csv(os.path.join(self.ckp_dir, 'state_history.csv'))
        
        
        self.logger.info('>'*30)
        self.logger.info('Gbest result: {:.4f}'.format(psb.history.gbest.fitness))            
        for i in range(self.opts.lag_order):
            Maskcode.append(psb.history.gbest.state[i])
            self.logger.info('stateMask_{}: {}'.format(i, int(psb.history.gbest.state[i])))
        
        return Maskcode
    
    def pack_data(self, data_set, model):
        data_loader = model.data_loader(data_set)
        h_states, x, y = model.batch_transform(data_loader)
        f_hidden = model.io_check(h_states, x)
        
        data_pack = Opt()
        data_pack.hidden_states = f_hidden.detach().clone()
        data_pack.target_seqs = y.detach().clone()
        
        return data_pack
    
    def state_selection(self,train_pack, val_pack):
        exp_cid = self.opts.cid
        # exp_cid = 0
        self.tuner_dir = os.path.join(self.host_dir, 'tuner', f'cid{exp_cid}')
        os_makedirs(self.tuner_dir)
        
        self.ckp_dir = os.path.join(self.host_dir, 'checkpoint', f'cid{exp_cid}')
        os_makedirs(self.ckp_dir)
        
        Maskcode_path = os.path.join(self.ckp_dir, 'series{}.cid{}.mask.pt'.format(self.opts.sid, exp_cid))
        model_state_path = os.path.join(self.ckp_dir, 'series{}.cid{}.model.pt'.format(self.opts.sid, self.opts.cid))
        
        if os.path.exists(Maskcode_path):
            self.logger.info(f'Loading maskcode from: {Maskcode_path}')
            Maskcode = torch.load(Maskcode_path)
        else:
            Maskcode = self.optimize(train_data=train_pack, valid_data=val_pack)
            torch.save(Maskcode, Maskcode_path)
        
            if os.path.exists(self.tuner_dir):
                os_rmdirs(self.tuner_dir) # to save space
                
        return model_state_path, Maskcode
        
        
    def xfit(self, train_data, valid_data, force_update = False):
        self.model = MaskESN(self.model_opts.hyper, self.logger)
        
        train_pack = self.pack_data(train_data, self.model)
        val_pack = self.pack_data(valid_data, self.model)
        
        model_state_path, Maskcode = self.state_selection(train_pack, val_pack)
            
        if os.path.exists(model_state_path) and force_update is False:
            checkpoint = torch.load(model_state_path)
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.init_stateMask(maskcode=Maskcode)
            self.logger.info(f'Loading model state from: {model_state_path}')
            
            self.fit_info = checkpoint['fit_info']
        else:
            self.model.init_stateMask(maskcode=Maskcode)
            self.fit_info = self.model.xfit(train_data, valid_data)
            checkpoint = {}
            checkpoint['model_state'] = self.model.state_dict()
            checkpoint['fit_info'] = self.fit_info
            torch.save(checkpoint, model_state_path)
        
        return self.fit_info
    
    def task_pred(self,task_data):
        x, y, pred = self.model.task_pred(task_data)
        return x, y, pred