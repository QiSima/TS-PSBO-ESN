import os, sys
from task.TaskLoader import TaskDataset, Opt
import numpy as np
import pandas as pd
        
class GEF2017ISO_Data(TaskDataset):
    def __init__(self, opts):
        super().__init__(opts)

    def info_config(self):
        self.info.normal = False
        self.info.lag_order = 24*7
        self.info.period = 24
        self.info.batch_size = 256
    def sub_config(self,):
        ts_tuple_list = []
        self.info.series_name = ['ISO NE CA']
        for i, name in enumerate(self.info.series_name):
            df = pd.read_excel('data/real/gef/2017_smd_hourly.xlsx',sheet_name=name, index_col=None,header=0)
            data = df['RT_Demand']
            if data.isnull().any():
                data= data.interpolate()
            raw_ts = data.values.reshape(-1, ) / 1000

            ts_tuple_list.append((name, raw_ts))
        
        self.pack_subs(ts_tuple_list)

class Sunspots_Data(TaskDataset):
    def __init__(self, opts):
        super().__init__(opts)

    def info_config(self):
        self.info.normal = False
        self.info.num_series = 1
        self.info.lag_order = 12*3
        self.info.period = 6
        self.info.batch_size = 512
        
    def sub_config(self,):
        ts_tuple_list = []
        self.info.series_name = ['Sunspots']
        
        for name in self.info.series_name:
            df = pd.read_csv('data/real/Sunspots/Sunspots.txt',header=None, index_col=None)
            raw_ts = df[0].values.reshape(-1,)
            ts_tuple_list.append((name, raw_ts))
        
        self.pack_subs(ts_tuple_list)