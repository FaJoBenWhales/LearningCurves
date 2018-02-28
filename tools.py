# Deep learning Theory final project

import os
import sys
import glob
import json
import datetime

import numpy as np
import pandas as pd
from sklearn import preprocessing

# global variables
seed = 42
'''
regul = {'lr_exp_decay' : False,
         'earlystop' : False, 
         'dropout' : False, 
         'L1L2' : False}
'''         

def get_run_name(prefix="run", additional=""):
    return "_".join([prefix, 
                     datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S"),
                     additional])


def configs_to_arr(configs):
    cfg_arr = np.zeros((len(configs), len(configs[0])))
    for i, config in enumerate(configs):
        for j, (key, value) in enumerate(configs[i].items()):
            cfg_arr[i][j] = value
            
    return cfg_arr


def load_data(source_dir='./data/final_project', timesteps = 5, scale_configs = True):
    
    data_dim = 1    
    configs = []
    learning_curves = []
    for fn in glob.glob(os.path.join(source_dir, "*.json")):
        with open(fn, 'r') as fh:
            tmp = json.load(fh)
            configs.append(tmp['config'])   # list of dicts
            learning_curves.append(tmp['learning_curve'])
    # print("loaded {} learning curves and configs".format(len(configs)))

    cfg_arr = configs_to_arr(configs)     # from list of dicts to np.array
    
    if scale_configs:
        print("scaling configuration data")
        cfg_arr = preprocessing.scale(cfg_arr)
   
    learning_curves = np.array(learning_curves)
    Y = learning_curves[:,-1] 

    lcs = learning_curves[:,:timesteps]
    # print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

    # Keras LSTM expects data as [sample_no, timesteps, feature_no (X.shape[1]) ]
    lcs = lcs.reshape(lcs.shape[0], timesteps, data_dim) 
    Y = Y.reshape(Y.shape[0],1)    

    return cfg_arr, lcs, Y


def load_lstm_data_concat_cfg(timesteps):
    
    configs, X, Y = load_data(scale_configs = True)
    
    X_enh = np.zeros((X.shape[0], timesteps, 1 + configs.shape[1]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_enh[i][j] = np.append(X[i][j][0],configs[i])

    return configs, X_enh, Y
