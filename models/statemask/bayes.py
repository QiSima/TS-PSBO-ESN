import os

import torch
import torch.nn as nn
import numpy as np

from ray.tune.search import ConcurrencyLimiter
# from ray.tune.search.ax import AxSearch
from ray.tune.search.optuna import OptunaSearch
import ray
from ray import tune
from ray.tune import ExperimentAnalysis
from ray.air import session, FailureConfig
from ray.air.config import RunConfig
from ray.air.config import CheckpointConfig

from models.statemask.util import Praticle2fitness
from task.TaskLoader import Opt


    
class bayesIndividual(Praticle2fitness, tune.Trainable):
    def setup(self, config, data = None) -> None:
        self.train_data = data.train_data
        self.valid_data = data.valid_data
        
        # self.hidden_states = data.hidden_states
        # self.target_seqs = data.target_seqs
        self.Reg_lambda = data.Reg_lambda if 'Reg_lambda' in data.dict else 0
        self.loss = nn.MSELoss()
        self.problem_dim = self.train_data.hidden_states.size(2)
        
        self.stateMask = [config['stateMask_{}'.format(i)] for i in range(self.problem_dim)]
        self.hiddens = self.state_select(self.train_data.hidden_states, self.stateMask)
        self.tgts = self.state_select(self.train_data.target_seqs, self.stateMask)

    def reset_config(self, new_config):        
        self.stateMask = [new_config['stateMask_{}'.format(i)] for i in range(self.problem_dim )]
        self.hiddens = self.state_select(self.train_data.hidden_states, self.stateMask)
        self.tgts = self.state_select(self.train_data.target_seqs, self.stateMask)
        
        return True
        
    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        torch.save(self.stateMask, checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        self.stateMask = torch.load(checkpoint_path)

    
class bayesTuner(Opt):
    def __init__(self, save_dir = None, data_pack = None, num_samples = 20, tuning_dict = None, metric = 'vmse', algo='tpe'):
        super().__init__()

            
        self.best_config = Opt()
        self.data_pack = data_pack
        
        self.metric = metric
        self.num_samples = num_samples
        self.dir = save_dir
        self.points_to_evaluate = []
        self.evaluated_rewards = []
        
        self.tuning_dict = tuning_dict
        self.algo = algo
        
    def gen_p2e(self, points_to_evaluate = None, evaluated_rewards = None):
        
        if evaluated_rewards is not None and points_to_evaluate is not None:
            assert len(points_to_evaluate) == len(evaluated_rewards)
            assert len(points_to_evaluate) > 0
            
            self.points_to_evaluate = points_to_evaluate
            self.evaluated_rewards = evaluated_rewards
    
    def conduct(self,):

        if self.algo == 'tpe':
            if len(self.points_to_evaluate) > 0:
                self.algo_func =  ConcurrencyLimiter(OptunaSearch(points_to_evaluate=self.points_to_evaluate, evaluated_rewards=self.evaluated_rewards),max_concurrent=6)
            else:
                self.algo_func = OptunaSearch(metric= self.metric, mode='min')
            self.algo_name = 'TPE_Search'
        elif self.algo == 'he':
            from ray.tune.search.hebo import HEBOSearch
            self.algo_func = HEBOSearch(metric= self.metric, mode='min')
            self.algo_name = 'HEBOSearch'
            # pip install 'HEBO>=0.2.0
            # @article{Cowen-Rivers2022-HEBO,
            # author = {Cowen-Rivers, Alexander and Lyu, Wenlong and Tutunov, Rasul and Wang, Zhi and Grosnit, Antoine and Griffiths, Ryan-Rhys and Maravel, Alexandre and Hao, Jianye and Wang, Jun and Peters, Jan and Bou Ammar, Haitham},
            # year = {2022},
            # month = {07},
            # pages = {},
            # title = {HEBO: Pushing The Limits of Sample-Efficient Hyperparameter Optimisation},
            # volume = {74},
            # journal = {Journal of Artificial Intelligence Research}
            # }
        elif self.algo == 'hb':
            from ray.tune.search.bohb import TuneBOHB
            from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
            self.algo_func = TuneBOHB(metric= self.metric, mode='min')
            self.algo_name = 'TuneBOHB'
            self.bohb_hyperband = HyperBandForBOHB(
                time_attr="training_iteration",
                max_t=1,
                reduction_factor=2,
                stop_last_trials=False,
            )
            # pip install dragonfly-opt
            # @article{JMLR:v21:18-223,
            # author  = {Kirthevasan Kandasamy and Karun Raju Vysyaraju and Willie Neiswanger and Biswajit Paria and Christopher R. Collins and Jeff Schneider and Barnabas Poczos and Eric P. Xing},
            # title   = {Tuning Hyperparameters without Grad Students: Scalable and Robust Bayesian Optimisation with Dragonfly},
            # journal = {Journal of Machine Learning Research},
            # year    = {2020},
            # volume  = {21},
            # number  = {81},
            # pages   = {1-27},
            # url     = {http://jmlr.org/papers/v21/18-223.html}
            # }        
        elif self.algo == 'bs':
            from flaml import BlendSearch
            self.algo_func = BlendSearch(metric= self.metric, mode='min', time_budget_s=3600)
            self.algo_name = 'BlendSearch'
            # @inproceedings{wang2021blendsearch,
            #     title={Economical Hyperparameter Optimization With Blended Search Strategy},
            #     author={Chi Wang and Qingyun Wu and Silu Huang and Amin Saied},
            #     year={2021},
            #     booktitle={ICLR'21},
            # }            
        else:
            raise ValueError(f'Non-supported algo : {self.algo}')
        
        self.tuning_results_path = os.path.join(self.dir, self.algo_name, 'search_results.pt')                
        
        if os.path.exists(self.tuning_results_path):
            save_dict = torch.load(self.tuning_results_path)
            config = save_dict['config']
            metrics = save_dict['metrics']
        else:
            config, metrics = self._conduct()
        
        return config, metrics
    
       
    def _conduct(self,):
        
        # ray.init(num_cpus=self.tuner.num_cpus)
        os.environ['RAY_COLOR_PREFIX'] = '1'
        ray.init()
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(bayesIndividual, data=self.data_pack), 
                resources={"gpu":1}),
            param_space=self.tuning_dict,
            tune_config=
            tune.TuneConfig(
            search_alg=self.algo_func,
            metric=self.metric,
            mode="min",
            num_samples=self.num_samples,
            reuse_actors=True,
            scheduler= self.bohb_hyperband if self.algo == 'hb' else None
            ),
            run_config=RunConfig(
                name=self.algo_name,
                storage_path=self.dir,
                verbose=1,
                failure_config=FailureConfig(max_failures=self.num_samples // 2),
                stop={'training_iteration':1},
                checkpoint_config=CheckpointConfig(
                    checkpoint_frequency=0,
                    checkpoint_at_end = False
                ),
                sync_config=tune.SyncConfig(syncer=None)
            )
        )
        
        results = tuner.fit() 
        df = results.get_dataframe()
        df.to_csv(os.path.join(self.dir, '{}.trial.csv'.format(self.algo_name)))
        ray.shutdown()
        
        best_result = results.get_best_result(self.metric, 'min')
        
        save_dict = {'config': best_result.config, 'metric':best_result.metrics[self.metric]}
        torch.save(save_dict, self.tuning_results_path)
        
        return best_result.config, best_result.metrics[self.metric]