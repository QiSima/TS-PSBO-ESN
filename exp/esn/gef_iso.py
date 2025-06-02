import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))

from task.TaskLoader import Opt
from task.TaskWrapper import Task
from task.parser import get_parser
from exp._model_config import esn_base
from exp._model_config import psb_esn_base
from data.realdata import GEF2017ISO_Data as Data
hyper_config={'readout_steps': 24*6,'iw_bound' : (-0.1, 0.1), 'weight_scaling':0.7, 'hidden_size':400, 'nonlinearity': 'tanh'}

class esn(esn_base):
    def task_modify(self):
        self.hyper.update(hyper_config)

class psb_esn(psb_esn_base):
    def task_modify(self):
        self.hyper.esn_opts.hyper.update(hyper_config)