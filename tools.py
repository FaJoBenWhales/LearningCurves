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

def scatter_plot_multi(y_true, figsize, results, title, steps=[]):

    fig = plt.figure(figsize=figsize)
    
    y_true = y_true.reshape(y_true.shape[0])
    
    for i, y_pred in enumerate(results['y_preds']):    

        mse = ((y_pred - y_true)**2).mean()
        # print("mses", mse, res['mse'])

        # fig = plt.figure(figsize=(10, 5))
        ax = plt.subplot(1, len(results), i+1)

        ax.set_xlim([0.13,1])
        ax.set_ylim([0.13,1])

        ax.set_xscale("log")
        ax.set_yscale("log")    
        ax.plot([0,1], [0,1], color='red')

        ax.scatter(y_true, y_pred)
    
        ax.set_title("{} {} steps, mse: {:.4f}".format(title,steps[i],mse))
        ax.set_xlabel("true final points")
        # ax.setp(ax.get_xticklabels(), visible=False)      
        
        if i==0:
            ax.set_ylabel("predicted final points")
    
    png_path = os.path.join("plots/", title+"_sct.png")    
    # print("path", png_path)
    fig.savefig(png_path)

    fig.show()
    

def scatter_plot(y_true, res, title, idx=0):
    
    from matplotlib.ticker import FormatStrFormatter    

    y_pred = res['y_preds'][idx]
    y_true = y_true.reshape(y_true.shape[0])
    
    mse = ((y_pred - y_true)**2).mean()
    # print("mses", mse, res['mse'])

    # fig = plt.figure(figsize=(10, 5))
    fig, ax = plt.subplots(figsize=(5, 5))    
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
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
    
    
    plt.title("model {}, mse: {:.5f}".format(title,mse))
    plt.xlabel("true final points")
    plt.ylabel("predicted final points")
    
    png_path = os.path.join("plots/", title+"_sct.png")    
    # print("path", png_path)
    plt.savefig(png_path)

    plt.show()


# results is list of dicts as returned by m.eval_cv, labels is list of according lables
def box_plot_single(y_true, figsize, results, labels, title, steps=None):

    fig = plt.figure(figsize=figsize)

    mses = []   # init of lists of mses
    for i, result in enumerate(results):
        y_pred = result['y_preds'][0]
        mses.append((y_pred - y_true.reshape(y_true.shape[0]))**2)

    ax = plt.subplot()
    ax.set_title(title)
    
    ax.set_ylim([0.00001,1])
    ax.set_yscale("log")     

    bplot = ax.boxplot(mses,vert=True, patch_artist=True, showmeans=True, meanline=True)
    plt.setp(ax, xticks=[y+1 for y in range(len(labels))],
             xticklabels=labels)

    png_path = os.path.join("plots/", title+"_box.png")    
    # print("path", png_path)
    fig.savefig(png_path)

    fig.show()    

    

# results is list of dicts as returned by m.eval_cv, labels is list of according lables
def table_plot(y_true, figsize, results, labels, title, steps=None):

    fig = plt.figure(figsize=figsize)

    mses = np.zeros((len(results), len(results[0]['y_preds'])))
    
    for i, result in enumerate(results):
        for j, y_pred in enumerate(result['y_preds']):
            mses[i][j] = ((y_pred - y_true.reshape(y_true.shape[0]))**2).mean()
    print("mses", mses)                

    table = [['%.2f' % j for j in i] for i in mses]    
    
    print("table", table)
    
    mses = np.round(mses,5)
    ax = plt.subplot()    
    # ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=mses,colLabels=labels,rowLabels=steps,loc='center')

    png_path = os.path.join("plots/", title+"_box.png")    
    fig.savefig(png_path)

    fig.show()    
    
    
    

# def extrapol_plot(cnn_after, x, ys, xlabel, ylabel, title=""):
def extrapol_plot(lc_true, lc_pred, step, idx, title="extrapolate"):
    
    """Create and save matplotlib plot with the desired data.
    ys is a dict of data lines with their labels as keys."""
    
    x = range(len(lc_true))    
    
    plt.figure()
    # for (ylabel, y) in ys.items():
    line_true = plt.plot(x,        lc_true,        label = "true learning curve", color = 'b')
    line_pred = plt.plot(x[step:], lc_pred[step:], label = "extrapolation", color = 'r')
    plt.title("rnn trained on random length \npredicting learning curve after step " + str(step))
    # plt.xlabel(xlabel)    
    # plt.legend([line_1,line_2], ['frozen cnn layers', 'unfreezing top 2 cnn layer-blocks'])
    plt.legend()
    # plt.legend([line_1,line_2], ['frozen cnn layers', 'gaga'])
    
    # plt.ylabel(ylabel)


    
    path = os.path.join("plots/", title+'_'+str(idx)+'_extra.png')
    plt.savefig(path)    
    
    plt.show()       
    

    
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
