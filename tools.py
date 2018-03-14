# Deep learning Theory final project

import os
import sys
import glob
import json
import datetime
import pickle

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
# res is dict as returned by m.eval_cv
def scatter_plot(y_true, res, title):

    y_pred = res['y_pred']
    
    y_true = y_true.reshape(y_true.shape[0])

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot()    
    
    # fig, ax = plt.subplots()
    # fit = np.polyfit(y_true, y_pred, deg=1)
    # ax.plot(y_true, fit[0] * y_true + fit[1], color='red')
    # plt.loglog(t, 20*np.exp(-t/10.0), basex=2)    
    # ax.loglog([0.0,0.95], [0.0,0.95], color='red')
    axes = plt.gca()
    axes.set_xlim([0.13,1])
    axes.set_ylim([0.13,1])
    
    ax.set_xscale("log")
    ax.set_yscale("log")    
    ax.plot([0,1], [0,1], color='red')
    ax.scatter(y_true, y_pred)
    
    plt.title("model {}, mse: {:.5f}".format(title,res['mse']))
    plt.xlabel("true final points")
    plt.ylabel("predicted final points")
    
    png_path = os.path.join("plots/", title+"_sct.png")    
    print("path", png_path)
    fig.savefig(png_path)

    fig.show()

# results is list of dicts as returned by m.eval_cv, labels is list of according lables
def box_plot(y_true, figsize, results, labels, title, steps=None):

    fig = plt.figure(figsize=figsize)
    
    for i, result in enumerate(results):
    
        mses = []   # init of lists of mses
        for j, y_pred in enumerate(result['y_preds']):
            mses.append((y_pred - y_true.reshape(y_true.shape[0]))**2)
            # print("this mse", labels[j], ((y_pred - y_true.reshape(y_true.shape[0]))**2).mean())
            # print("this mse from list", labels[j], mses[-1].mean())

        if len(results)>1:
            ax = plt.subplot(1, len(results), i+1)
            ax.set_title(title + str(steps[i]))
        else:
            ax = plt.subplot()
            ax.set_title(title)
    
        ax.set_ylim([0.00001,1])
        ax.set_yscale("log")     


        bplot = ax.boxplot(mses,vert=True, patch_artist=True, showmeans=True, meanline=True)

        plt.setp(ax, xticks=[y+1 for y in range(len(labels))],
                 xticklabels=labels)

    png_path = os.path.join("plots/", title+"_sct.png")    
    print("path", png_path)
    fig.savefig(png_path)

    fig.show()


def pickle_from_file(fname):
    
    path = os.path.join("plots/", fname+".pickle")      
    with open(path, 'rb') as handle:
        obj = pickle.load(handle)
        
    return obj
    
    
def pickle_to_file(obj, fname):
    
    path = os.path.join("plots/", fname+".pickle")      
    
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

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
