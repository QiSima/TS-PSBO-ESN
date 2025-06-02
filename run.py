import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))

from task.TaskLoader import Opt
from task.TaskWrapper import Task
from task.parser import get_parser
               
if __name__ == "__main__":

    parser = get_parser(parsing=False)
    args = parser.parse_args()
    
    args.cuda = True
    args.datafolder = 'exp/esn'

    args.exp_name = 'esn'

    # 实验数据集1
    args.dataset = 'lz'
    args.H = 1

    # # # 实验数据集2
    # args.dataset = 'sun'
    # args.H = 1

    # # # 实验数据集3
    # args.dataset = 'gef_iso'
    # args.H = 24

    args.model = 'psb_esn'
    
    args.rep_times = 10

    task = Task(args)
    task.conduct()
    args.metrics = ['rmse','smape', 'nrmse']
    task.evaluation(args.metrics)    