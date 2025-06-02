from task.TaskLoader import Opt
from ray import tune

class nn_base(Opt):
    def __init__(self):
        super().__init__()
        self.hyper = Opt()
        
        self.tuner = Opt()
        self.tuning = Opt()

        self.hyper_init()
        self.tuner_init()

        self.base_modify()
        self.hyper_modify()

        self.tuning_modify()
        self.ablation_modify()
        self.task_modify()
        
        self.common_process()

    def hyper_init(self,):
        pass

    def tuner_init(self,):
        # total cpu cores for tuning
        self.tuner.resource = {
            "cpu": 5,
            "gpu": 0.5  # set this for GPUs
        }
        # gpu cards per trial in tune
        # tuner search times
        self.tuner.num_samples = 20
        # fitness epoch per iter
        self.tuner.epochPerIter = 1
        # self.tuner.algo = 'rand'

    def base_modify(self,):
        pass

    def hyper_modify(self):
        pass

    def tuning_modify(self):
        pass

    def ablation_modify(self):
        pass

    def task_modify(self):
        pass

    def common_process(self,):
        if "import_path" in self.dict:
            self.import_path = self.import_path.replace(
            '.py', '').replace('/', '.')
            
class esn_base(nn_base):
    def __init__(self):
        self.training = False
        self.arch = 'rnn'
        super().__init__()
        
    def base_modify(self,):
        self.import_path='models/stochastic/ESN.py'
        self.class_name = 'EchoStateNetwork'
        
    def hyper_init(self,):        
        self.hyper.leaky_r = 1
        self.hyper.readout_steps = 1 # last states, default 1 (equal to FCD output arch.)
        self.hyper.hidden_size = 400
        self.hyper.reg_lambda = 0
        self.hyper.nonlinearity = 'tanh'
        self.hyper.iw_bound = (-0.1, 0.1)
        self.hyper.hw_bound = (-1, 1)
        self.hyper.weight_scaling = 0.9
        self.hyper.init = 'vanilla'
        self.hyper.fc_io = 'step'
        self.hyper.input_dim = 1
    
    def tuner_init(self):
        self.tuner.resource = {
            "cpu": 10,
            "gpu": 1  # set this for GPUs
        }
        self.tuner.algo = 'grid'
        self.tuner.num_samples = 1
        self.tuning.iw_bound = tune.grid_search([0.0001, 0.001, 0.01, 0.1])
        self.tuning.weight_scaling = tune.grid_search([i * 0.1 for i in range(2, 10)])
        self.tuning.hidden_size = tune.grid_search([100, 200, 300, 400, 500])
    
class psb_esn_base(nn_base):
    def __init__(self):
        super().__init__()
        self.training = False
        self.arch = 'rnn'
        self.innerTuning = True
            
    def base_modify(self,):
        self.import_path = 'models/stochastic/psbESN.py'
        self.class_name = 'PSB_ESN'
        
    def hyper_init(self):
        self.hyper.esn_opts=esn_base()
        self.hyper.esn_opts.import_path = 'models/statemask/MESN.py'
        self.hyper.esn_opts.class_name = 'MaskESN'
        self.hyper.esn_opts.common_process()
                
        self.hyper.pop_size = 100
        self.hyper.maxCycle = 100