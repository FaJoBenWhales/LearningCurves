# Deep learning Theory final project

import os
import sys
import glob
import json
import datetime

import numpy as np
import pandas as pd
from sklearn import preprocessing

'''
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat
'''

# global variables
seed = 42

# workaround: passing these as arguments to the simle_mlp() wrapper function causes an error in keras 
# (a well-known bug of Tensorflow) so they have to be held in this global dictionary
mlp_cfg ={'lr': 0.1,
          'input_dim' : 1,
          'dropout' : False,
          'L1L2' : False,
          'earlystop' : True, 
          'lr_exp_decay' : False,
          'k_exp' : 0.1}

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


def load_data(source_dir='./data/final_project', scale_cfgs = False):
    configs = []
    learning_curves = []
    for fn in glob.glob(os.path.join(source_dir, "*.json")):
        with open(fn, 'r') as fh:
            tmp = json.load(fh)
            configs.append(tmp['config'])   # list of dicts
            learning_curves.append(tmp['learning_curve'])
    # print("loaded {} learning curves and configs".format(len(configs)))

    cfg_arr = configs_to_arr(configs)     # from list of dicts to np.array
    
    if scale_cfgs:
        cfg_arr = preprocessing.scale(cfg_arr)
   
    learning_curves = np.array(learning_curves)
    Y = learning_curves[:,-1] 

    return cfg_arr, Y, learning_curves


def load_lstm_data(timesteps):
    
    data_dim = 1
    mlp_cfg['input_dim'] = data_dim
    cfgs, lcs_end, LCS = load_data(scale_cfgs = True)
    # take
    
    lcs = LCS[:,:timesteps]
    # print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

    # Keras LSTM expects data as [sample_no, timesteps, feature_no (X.shape[1]) ]
    lcs = lcs.reshape(lcs.shape[0], timesteps, data_dim) 
    lcs_end = lcs_end.reshape(lcs_end.shape[0],1)

    return cfgs, lcs, lcs_end


def load_lstm_data_concat_cfg(timesteps):
    
    cfgs, Y, LCS = load_data(scale_cfgs = True)
    X = LCS[:,:timesteps]    
    
    data_dim = 1 + cfgs.shape[1]
    mlp_cfg['input_dim'] = data_dim
    # print("data dim", data_dim)
    
    # Keras LSTM expects data as [sample_no, timesteps, feature_no (X.shape[1]) ]
    X = X.reshape(X.shape[0], timesteps, 1) 
    Y = Y.reshape(Y.shape[0],1)    
    
    X_enh = np.zeros((X.shape[0], timesteps, 1 + cfgs.shape[1]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # print(X.shape, cfgs.shape)
            X_enh[i][j] = np.append(X[i][j][0],cfgs[i])
            gaga = 2
    # print(X.shape)
    # print(X[0])
    # print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)



    return cfgs, X_enh, Y



'''
def load_lstm_data(timesteps):
    
    data_dim = 1
    
    cfgs, Y, LCS = load_data(scale_cfgs = True)
    # take
    
    x_train, x_val = LCS[:,:timesteps][:200], LCS[:,:timesteps][200:]
    y_train, y_val = Y[:200], Y[200:]
    # print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

    # Keras LSTM expects data as [sample_no, timesteps, feature_no (X.shape[1]) ]
    x_train = x_train.reshape(x_train.shape[0], timesteps, data_dim) 
    x_val = x_val.reshape(x_val.shape[0], timesteps, data_dim) 
    y_train = y_train.reshape(y_train.shape[0],1)
    y_val = y_val.reshape(y_val.shape[0],1)

    return x_train, x_val, y_train, y_val
'''
 
# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


def scale(train, test):
    # fit scaler
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    
    return scaler, train_scaled, test_scaled

def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]
'''
# scale dataset into -1 / 1 before passing to LSTM
def minmax_scale(X):
    X = X.reshape(len(X), 1)
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(X)
    return scaler, scaler.transform(X)

def minmax_scale_back(scaler, X_scaled):
    return scaler.inverse_transform(X_scaled)
'''


# load dataset
#def parser(x):
#    return pd.datetime.strptime('190'+x, '%Y-%m')


#def test_fn()
#    series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    # transform to supervised learning
#    X = series.values
#    supervised = timeseries_to_supervised(X, 1)
#    print(supervised.head())
