import torch
import torch.nn as nn
import numpy as np
from task.TaskLoader import Opt
from models.statemask.util import Praticle2fitness,sigmoid_act,state2config
from models.statemask.bayes import bayesTuner
from numpy.random import uniform
from tqdm import trange, tqdm
import copy
from scipy.special import softmax

import os
from task.util import os_makedirs
import pandas as pd

class Praticle:
    def __init__(self, dim):
        self.dim = dim
        self.speed_u = 4
        self.speed_l = -4
        
        self.speed = np.zeros(dim)
        self.state = np.zeros(dim)
        # self.fitness= np.ones((pop_size,))

    def speed_bound(self, ):
        for d in range(self.dim):
            if self.speed[d] >= self.speed_u:
                self.speed[d] = self.speed_u
            if self.speed[d] <= self.speed_l:
                self.speed[d] = self.speed_l
    
    def state_init(self, state = None):
        # for i in range(self.pop_size):
        if state is not None:
            self.state = state
        else:
            self.state_update()
    
    def state_update(self,):
        for d in range(self.dim):
            self.state[d] = sigmoid_act(self.speed[d])

    
class History:
    def __init__(self, maxCycle, pop_size, dim):
        # pop_size, dim = pop.pop_size, pop.dim
        # self.speed_history = np.zeros((maxCycle, pop_size, dim))
        self.state_history = np.zeros((maxCycle, pop_size, dim))
        self.fitness_history = np.zeros((maxCycle, pop_size))
        self.config_history = []
        
        for t in range(maxCycle):
            for i in range(pop_size):
                self.fitness_history[t, i] = float('inf')

        self.pbest = []
        for i in range(pop_size):
            pci = Opt()
            pci.fitness = float('inf')
            pci.state = np.ones(dim)
            pci.history = np.zeros(maxCycle)
            self.pbest.append(pci)
            
            self.config_history.append([])
            
        
        self.gbest = Opt()
        self.gbest.fitness = float('inf')
        self.gbest.state = np.ones(dim)
        self.gbest.history = np.zeros(maxCycle)
        self.gbest.state_history = np.ones((maxCycle, dim))


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, logger, patience=30, delta=0):
        """
        Args:
            logger: log the info to a .txt
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_mse_min = np.Inf
        self.delta = delta
        self.logger = logger

    def __call__(self, score):

        if self.best_score is None:
            self.best_score = score
            self.logger.info(
                f'Validation accuracy increased ({self.val_mse_min:.3f}% --> {score:.3f})%.')
            self.val_mse_min = score
        elif score > self.best_score + self.delta:
            self.counter += 1
            self.logger.info(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.logger.info(
                f'Validation accuracy increased ({self.val_mse_min:.3f}% --> {score:.3f})%.')
            self.val_mse_min = score
            self.counter = 0
            
    
class PSB():
    def __init__(self, pop_size = 100, maxCycle = 100, bayes = True, bs_epochs= 10, train_data = None, valid_data = None, w_max = 0.9, w_min=0.4, c1=2.0, c2=2.0, save_dir = None, tuning_dict = None, metric = 'vmse', init_file='', top_k = 20, logger = None, patience = 30):
        self.pop_size = pop_size
        self.maxCycle = maxCycle
        self.bayes_tag = bayes
        
        self.loss_fn = nn.MSELoss()
        
        # init data
        self.train_data = train_data
        self.valid_data = valid_data
        self.problem_dims = len(list(tuning_dict.keys()))
        
        # init pop
        # self.pop = 
        self.pop = []
        for i in range(self.pop_size):
            p = Praticle(dim=self.problem_dims)
            self.pop.append(p)
        
        self.history = History(self.maxCycle, pop_size, self.problem_dims)
        
        self.data_pack = Opt()
        self.data_pack.train_data = train_data
        self.data_pack.valid_data = valid_data
        
        self.peval = Praticle2fitness()
        self.peval.setup(data=self.data_pack)
        
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.tuner_dir = save_dir
        self.tuning_dict = tuning_dict
        self.bs_epochs = bs_epochs
        self.fitness_metric = metric
        self.init_file = init_file
        self.top_k = top_k
        self.logger = logger
        self.earlystop =EarlyStopping(logger=self.logger, patience=patience)
        
    def cal_fitness(self, config):
        self.peval.reset_config(config)
        results = self.peval.step()
        fitness = results[self.fitness_metric]
        return fitness
    
    def bayes_search(self, pop_id=0, cycle_id = 0):
        i = pop_id
        i_history = self.history.config_history[i]
        points = len(i_history)
        i_rewards = self.history.fitness_history[:points, i].tolist()
        
        # fint top_k rewards and their ids
        res = sorted(range(len(i_rewards)), key=lambda sub: i_rewards[sub])[:self.top_k]
        points_to_evaluate =[]
        evaluated_rewards = []
        res.reverse()
        for k in res:
            points_to_evaluate.append(i_history[k])
            evaluated_rewards.append(i_rewards[k])
            
        
        bo_dir = os.path.join(self.tuner_dir, f'iter{cycle_id}.pop{pop_id}')
        os_makedirs(bo_dir)
        
        bo = bayesTuner(save_dir=bo_dir, data_pack=self.data_pack, tuning_dict=self.tuning_dict, num_samples=self.bs_epochs, metric = self.fitness_metric)
        bo.gen_p2e(points_to_evaluate, evaluated_rewards)
        
        bo_config, bo_fitness = bo.conduct()
        bo_state = np.zeros(self.problem_dims)
        for d in range(self.problem_dims):
            bo_state[d] = int(bo_config['stateMask_{}'.format(d)])
            
        return bo_config, bo_state, bo_fitness
    
    def pop_init(self,):
        ef_list = []
        if os.path.exists(self.init_file):
            df = pd.read_csv(self.init_file, header=0, index_col=0)
            df_nums =  df.shape[0]
            df_vmse = []
            for trail in range(df_nums):             
                ef_vmse = df.iloc[trail]['vmse']
                df_vmse.append(ef_vmse)
            
            # df_min = min(df_vmse)
            nums = min(self.pop_size, df.shape[0])
            
            res = sorted(range(len(df_vmse)), key=lambda sub: df_vmse[sub])[:nums]
            
            ef_list = []
            for k in res:
                # ef_fintess= df_vmse[k]
                ef_state = np.zeros(self.problem_dims)
                for t in range(self.problem_dims):
                    ef_state[t] = float(df.iloc[k]['config/stateMask_{}'.format(t)])
                
                ef_list.append(ef_state)
            
            print('>'*20)
            print('Initial successfully!')
            print('>'*20)
        return ef_list
            
                    
            
    
    def evolve(self,):
        # grid_interval = 3
        # read_id_list = list(range(1, self.problem_dims, grid_interval)) # limited to only state selection
        # ef_id_list = read_id_list[:self.pop_size]
        # if len(read_id_list) < self.pop_size:
        #     ef_id_list = read_id_list
        
        # last_fitness = np.zeros(self.pop_size)
        ef_list = self.pop_init()
        gbest_state = np.zeros(self.problem_dims)
        for t in trange(self.maxCycle, colour='green'):
            for i in trange(self.pop_size, colour='red', leave=False):
                if t == 0:
                    # init the pbest and gbest 
                    if i < len(ef_list):
                        ef_state = ef_list[i]
                        self.pop[i].state_init(state= ef_state)
                    else:
                        self.pop[i].state_init()
                    
                else:
                    w_t = (self.w_max- self.w_min) * (self.maxCycle - t + 1)/(t+1) + self.w_min
                    
                    rand1 = uniform(0,1)
                    rand2 = uniform(0,1)
                    
                    speed_t = w_t * self.pop[i].speed + self.c1* rand1 * (self.history.pbest[i].state - self.pop[i].state) + self.c2 * rand2 * (self.history.gbest.state - self.pop[i].state)
                    self.pop[i].speed = speed_t
                    
                    self.pop[i].speed_bound()
                    self.pop[i].state_update()
                    
                config = state2config(self.pop[i].state)
                fitness = self.cal_fitness(config)
                self.history.fitness_history[t, i] = fitness
                self.history.config_history[i].append(config)
                    
                if fitness < self.history.pbest[i].fitness:
                    self.history.pbest[i].fitness = fitness
                    self.history.pbest[i].state = copy.deepcopy(self.pop[i].state)
                    self.logger.info(f'Update pbest of pop {i} in the iteration {t}:')
                    self.logger.info('{}'.format(self.history.pbest[i].state.astype('int')))
                    
                    if fitness < self.history.gbest.fitness:
                        self.history.gbest.fitness = fitness
                        self.history.gbest.state = copy.deepcopy(self.pop[i].state)
                        gbest_state = copy.deepcopy(self.history.gbest.state)
                        self.logger.info('>'*60)
                        self.logger.info(f'Update gbest of pop {i} in the iteration {t}:')
                        self.logger.info('{}'.format(self.history.gbest.state.astype('int')))                        
                    else:
                        if not np.array_equal(gbest_state.astype('int'), self.history.gbest.state.astype('int')):
                            if t > 0:
                                self.logger.info('!'*30)
                                self.logger.info(f'Wrong in memorizing Gbest state with iteration {t} on pop {i}:')
                                self.logger.info('{}'.format(self.history.gbest.state.astype('int')))
                                self.logger.info('Saved Gbest state:\n{}'.format(gbest_state.astype('int')))
                                raise ValueError(f'Wrong in memorizing Gbest state with iteration {t} on pop {i}')
                

                self.history.pbest[i].history[t] = self.history.pbest[i].fitness
            
            if self.bayes_tag and t >= self.top_k:
                cur_fitness = self.history.fitness_history[t,:]
                worst_fitness = cur_fitness.max()
                p_ls = softmax(np.ones_like(cur_fitness)* worst_fitness - cur_fitness)
                
                for i in trange(self.pop_size):
                    roll = uniform(0, 1)
                    if p_ls[i] >= roll:
                        bo_config, bo_state, bo_fitness = self.bayes_search(pop_id=i, cycle_id = t)
                        if bo_fitness < self.history.fitness_history[t,i]:
                            self.pop[i].state = copy.deepcopy(bo_state)
                            self.history.config_history[i][-1] = copy.deepcopy(bo_config)
                            self.history.fitness_history[t, i] = bo_fitness
                        
                            if bo_fitness < self.history.pbest[i].fitness:
                                self.history.pbest[i].fitness = bo_fitness
                                self.history.pbest[i].state = copy.deepcopy(self.pop[i].state)
                                self.logger.info(f'BO Update pbest of pop {i} in the iteration {t}:')
                                self.logger.info('{}'.format(self.history.pbest[i].state.astype('int')))
                                                    
                                if bo_fitness < self.history.gbest.fitness:
                                    self.history.gbest.fitness = bo_fitness
                                    self.history.gbest.state = copy.deepcopy(self.pop[i].state)
                                    self.logger.info('>'*60)
                                    self.logger.info(f'BO Update gbest of pop {i} in the iteration {t}:')
                                    self.logger.info('{}'.format(self.history.gbest.state.astype('int')))                                      

                            self.history.pbest[i].history[t] = self.history.pbest[i].fitness

            self.history.gbest.history[t] = self.history.gbest.fitness
            self.history.gbest.state_history[t,:] = self.history.gbest.state.astype('int')
            gbest_state = copy.deepcopy(self.history.gbest.state)
            
            current_igbest = self.history.fitness_history[t, :].min()
            self.earlystop(current_igbest)
            if self.earlystop.early_stop:
                self.logger.info('Early stopping')
                break
            
class PO():
    def __init__(self, pop_size = 100, maxCycle = 100, train_data = None, valid_data = None, save_dir = None, tuning_dict = None, metric = 'vmse', logger = None, patience = 30, algo = 'clpso'):
        self.pop_size = pop_size
        self.maxCycle = maxCycle
        
        self.loss_fn = nn.MSELoss()
        
        # init data
        self.train_data = train_data
        self.valid_data = valid_data
        self.problem_dims = len(list(tuning_dict.keys()))
        
        self.data_pack = Opt()
        self.data_pack.train_data = train_data
        self.data_pack.valid_data = valid_data
        
        self.peval = Praticle2fitness()
        self.peval.setup(data=self.data_pack)
        
        self.tuner_dir = save_dir
        self.tuning_dict = tuning_dict

        self.fitness_metric = metric
        self.logger = logger
        self.patience = patience       
        self.algo = algo  
    
    
    def evolve(self,):
        def fitness_function(solution):
            config = {}
            for i in range(self.problem_dims):
                state_i = solution[i]
                rand = uniform(0, 1)
                if state_i > rand:
                    config['stateMask_{}'.format(i)] = 1
                else:
                    config['stateMask_{}'.format(i)] = 0
                
            self.peval.reset_config(config)
            results = self.peval.step()
            fitness = results[self.fitness_metric]
            
            return fitness

        problem = {
            "fit_func": fitness_function,
            "lb": [0 for i in range(self.problem_dims)],
            "ub": [1 for i in range(self.problem_dims)],
            "minmax": "min",
            'verbose': True
        }
        
        term_dict = {
        "max_early_stop": self.patience   # after 30 epochs, if the global best doesn't improve then we stop the program
        }
        
        if self.algo == 'clpso':
            from mealpy.swarm_based import PSO
            algo_func = PSO.CL_PSO(epoch=self.maxCycle,pop_size=self.pop_size)
            # Liang, J.J., Qin, A.K., Suganthan, P.N. and Baskar, S., 2006. Comprehensive learning particle swarm optimizer for global optimization of multimodal functions. IEEE transactions on evolutionary computation, 10(3), pp.281-295.
        elif self.algo == 'aro':
            from mealpy.swarm_based import ARO
            algo_func = ARO.OriginalARO(epoch=self.maxCycle,pop_size=self.pop_size)
            # Wang, L., Cao, Q., Zhang, Z., Mirjalili, S., & Zhao, W. (2022). Artificial rabbits optimization: A new bio-inspired meta-heuristic algorithm for solving engineering optimization problems. Engineering Applications of Artificial Intelligence, 114, 105082.
        
        algo_func.solve(problem, termination=term_dict)
        best_solution = algo_func.solution[0]
        Maskcode = []
        for i in range(self.problem_dims):
            state_i = best_solution[i]
            rand = uniform(0, 1)
            if state_i > rand:
                cur =  1
            else:
                cur = 0
            
            Maskcode.append(cur)
        
        return Maskcode        
        