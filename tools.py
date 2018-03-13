# Deep learning Theory final project

import os
import sys
import glob
import json
import datetime

import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

# global variables
seed = 42
'''
regul = {'lr_exp_decay' : False,
         'earlystop' : False, 
         'dropout' : False, 
         'L1L2' : False}
'''         

def scatter_plot(y_true, y_pred, mse):

    y_true = y_true.reshape(y_true.shape[0])
    fig, ax = plt.subplots()
    fit = np.polyfit(y_true, y_pred, deg=1)
    ax.plot(y_true, fit[0] * y_true + fit[1], color='red')
    ax.scatter(y_true, y_pred)
    
    plt.title("final points - mse: {:.5f}".format(mse))
    plt.xlabel("true final points")
    plt.ylabel("predicted final points")    

    fig.show()



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


def load_data(source_dir='./data/final_project', scale_configs = True):
    
    data_dim = 1    
    configs = []
    learning_curves = []
    for fn in glob.glob(os.path.join(source_dir, "*.json")):
        with open(fn, 'r') as fh:
            tmp = json.load(fh)
            configs.append(tmp['config'])   # list of dicts
            learning_curves.append(tmp['learning_curve'])

    configs = configs_to_arr(configs)     # from list of dicts to np.array
    
    if scale_configs:
        print("scaling configuration data")
        configs = preprocessing.scale(configs)
   
    lcs = np.array(learning_curves)
    Y = lcs[:,-1] 

    # Keras LSTM expects data as [sample_no, timesteps, feature_no (X.shape[1]) ]
    lcs = lcs.reshape(lcs.shape[0], lcs.shape[1], data_dim) 
    Y = Y.reshape(Y.shape[0],1)    

    return configs, lcs, Y


def load_lstm_data_concat_cfg(timesteps):
    
    configs, X, Y = load_data(scale_configs = True)
    
    X_enh = np.zeros((X.shape[0], timesteps, 1 + configs.shape[1]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_enh[i][j] = np.append(X[i][j][0],configs[i])

    return configs, X_enh, Y
