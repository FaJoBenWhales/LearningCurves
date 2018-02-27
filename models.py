# Deep learning Theory final project
import numpy as np
import time
from keras import optimizers
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, LSTM, concatenate
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb

import tools as t


def mlp():
    
    if t.mlp_cfg['L1L2']:
        kernel_regularizer=l1_l2(l1=0.01, l2=0.01)
        print("L1L2 regularisation")
    else: 
        kernel_regularizer = None

    model = Sequential()
    model.add(Dense(64, input_dim=5, kernel_initializer='normal', 
                    kernel_regularizer=kernel_regularizer, activation='relu'))
    if t.mlp_cfg['dropout']:
        model.add(Dropout(0.7))
    model.add(Dense(64, kernel_initializer='normal', 
                    kernel_regularizer=kernel_regularizer, activation='relu'))
    if t.mlp_cfg['dropout']:
        model.add(Dropout(0.7))
    model.add(Dense(1, kernel_initializer='normal'))
    
    opt = optimizers.SGD(lr=t.mlp_cfg['lr'])
    # opt = optimizers.Adam(lr=t.mlp_cfg['lr'], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)    
    model.compile(loss='mean_squared_error', optimizer=opt)  # no adaptive learning rate (--> later compare to exponential)
    
    return model


def lstm():
    data_dim = t.mlp_cfg['input_dim']      # features of input data
    # timesteps = 10    # 
    output_dim = 1
    
    model = Sequential()
    model.add(LSTM(64, return_sequences=True,
                   # input shape = (timesteps, data_dim) - but timesteps can be None for flexible use
                   input_shape=(None, data_dim)))  # returns a sequence of vectors of dimension 32
    # model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(64))  # return a single vector of dimension 32
    # model.add(Dense(10, activation='softmax'))
    model.add(Dense(64))
    model.add(Dense(64))
    model.add(Dense(1))    

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mse'])
    return model


def multi_lstm():
    
    data_dim = t.mlp_cfg['input_dim']      # features of input data, variable is set in load_lstm_data
    # timesteps = 10    # 
    output_dim = 1
    
    # model = Sequential()
    
    lcs_input = Input(shape=(None, data_dim), name='lcs_input')

    lstm_out = LSTM(64, return_sequences=True)(lcs_input)
    lstm_out = LSTM(64)(lstm_out)

    cfgs_input = Input(shape=(5,), name='cfgs_input')
    x = concatenate([cfgs_input, lstm_out])    
    
    x = Dense(64)(x)
    x = Dense(64)(x)
    main_output = Dense(1, name='main_output')(x)
    
    model = Model(inputs=[cfgs_input, lcs_input], outputs=[main_output])
    
    model.compile(loss='mean_squared_error',
                  optimizer='rmsprop',
                  metrics=['mse'])
    return model

'''
ToDo: pass learning rate at creation of model
def train_mlp(model, cfgs, Y, cfg, split, epochs):
    cfgs_train, cfgs_val = cfgs[:split], cfgs[split:]
    y_train, y_val = Y[:split], Y[split:]
    print(cfgs_train.shape, cfgs_val.shape, y_train.shape, y_val.shape)    

    model.fit(cfgs_train, y_train,
              batch_size=cfg['batch_size'], 
              epochs=epochs,
              validation_data=(cfgs_val, y_val))
'''


def train_lstm(model, lcs, Y, split, batch_size, epochs):
    x_train, x_val = lcs[:split], lcs[split:]
    y_train, y_val = Y[:split], Y[split:]
    print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)    

    model.fit(x_train, y_train,
              batch_size=batch_size, epochs=epochs,
              validation_data=(x_val, y_val))

    
def train_multi_lstm_old(model, cfgs, lcs, Y, split, batch_size, epochs):
    x_train, x_val = lcs[:split], lcs[split:]
    y_train, y_val = Y[:split], Y[split:]

    print(x_train.shape, x_val.shape, cfgs_train.shape, cfgs_val.shape, y_train.shape, y_val.shape)    

    model.fit([cfgs_train, x_train], y_train,
              batch_size=batch_size, epochs=epochs,
              validation_data=([cfgs_val, x_val], y_val))

    
def train_multi_lstm(model, x_cfgs, Y, split, batch_size, epochs):
    x_cfgs_train, x_cfgs_val = [], []

    # x_cfgs_train = list(len(x_cfgs))
    for i in range(len(x_cfgs)):
        x_cfgs_train.append(x_cfgs[i][:split])
        x_cfgs_val.append(x_cfgs[i][split:])
        
        # x_cfgs_train[i], x_cfgs_val[i] = x_cfgs[i][:split], x_cfgs[i][split:]
    
    # x_train, x_val = lcs[:split], lcs[split:]
    y_train, y_val = Y[:split], Y[split:]
    # cfgs_train, cfgs_val = cfgs[:split], cfgs[split:]
    print(len(x_cfgs_train), len(x_cfgs_val), y_train.shape, y_val.shape)    
    print("cfgs_train/val:", x_cfgs_train[0].shape, x_cfgs_val[0].shape, 
          "lcs_train/val:", x_cfgs_train[1].shape, x_cfgs_val[1].shape, 
          "lcs_end train/val:", y_train.shape, y_val.shape)    
    # print("x_cfgs_train/val:", x_cfgs_train.shape)#, x_cfgs_val.shape)
    
    # x_arr_train = np.array(x_cfgs_train)
    # x_arr_val = np.array(x_cfgs_val)
    
    model.fit(x_cfgs_train, y_train,
              batch_size=batch_size, epochs=epochs,
              validation_data=(x_cfgs_val, y_val))

def train_multi_lstm_test(model, x_cfgs, Y, split, batch_size, epochs):
    x_cfgs_train, x_cfgs_val = [], []
    
    for i in range(split):
        x_cfgs_train.append((x_cfgs[0][i],x_cfgs[1][i]))
    print("len(x_cfgs)", len(x_cfgs[0]))
    for i in range(split,len(x_cfgs[0])):
        x_cfgs_val.append((x_cfgs[0][i],x_cfgs[1][i]))
    
    '''
    # x_cfgs_train = list(len(x_cfgs))
    for i in range(len(x_cfgs)):
        x_cfgs_train.append(x_cfgs[i][:split])
        x_cfgs_val.append(x_cfgs[i][split:])
        
        # x_cfgs_train[i], x_cfgs_val[i] = x_cfgs[i][:split], x_cfgs[i][split:]
    '''
    
    # x_train, x_val = lcs[:split], lcs[split:]
    y_train, y_val = Y[:split], Y[split:]
    # cfgs_train, cfgs_val = cfgs[:split], cfgs[split:]
    print(len(x_cfgs_train), len(x_cfgs_val), y_train.shape, y_val.shape)    
    print("cfgs_train/val:", len(x_cfgs_train), len(x_cfgs_val)) 
    print("lcs_train/val[7]:", len(x_cfgs_train[7]), len(x_cfgs_val[7])) 
    print("lcs_train/val[7][0]:", len(x_cfgs_train[7][0]), len(x_cfgs_val[7][0]))
    print("lcs_train/val[7][0].shape:", x_cfgs_train[7][0].shape, x_cfgs_val[7][0].shape)    
    print("lcs_train/val[7][1].shape:", x_cfgs_train[7][1].shape, x_cfgs_val[7][1].shape)     
    print("lcs_end train/val:", y_train.shape, y_val.shape)   
    # print("x_cfgs_train/val:", x_cfgs_train.shape)#, x_cfgs_val.shape)
    
    # x_arr_train = np.array(x_cfgs_train)
    # x_arr_val = np.array(x_cfgs_val)
    
    model.fit(x_cfgs_train, y_train,
              batch_size=batch_size, epochs=epochs,
              validation_data=(x_cfgs_val, y_val))    
    
    
def eval_lstm(model, lcs, Y, split, batch_size):
    x_train, x_val = lcs[:split], lcs[split:]
    y_train, y_val = Y[:split], Y[split:]              
    print("x_train.shape",x_train.shape)
    print("x_train[0]",x_train[0])
    
    score, mse = model.evaluate(x_val, y_val,
                                batch_size=5)
    print("x_train.shape",x_train.shape)
    print("mse: ", mse)
    
    return mse


def eval_multi_lstm(model, cfgs, lcs, Y, split, batch_size):
    x_train, x_val = lcs[:split], lcs[split:]
    y_train, y_val = Y[:split], Y[split:]
    cfgs_train, cfgs_val = cfgs[:split], cfgs[split:]    
    print("x_val.shape",x_train.shape)
    print("x_val[0]",x_train[0])
    print("cfgs_val.shape",x_train.shape)
    print("cfgs_val[0]",x_train[0])
    
    score, mse = model.evaluate([x_val, cfgs_val], y_val,
                                batch_size=5)
    
    return mse

'''
def forecast_lstm(model, batch_size, lcs):
    lcs = lcs.reshape(1, 1, len(lcs))
    yhat = model.predict(lcs, batch_size=batch_size)
    return yhat[0,0]
'''


def exp_decay(epoch):
    initial_lrate = t.mlp_cfg['lr']
    k = t.mlp_cfg['k_exp']
    lrate = initial_lrate * np.exp(-k*epoch)
    if (epoch % 50 == 0):
        print("new lr: ", lrate)
    return lrate



def get_estimator(model_type, X = None, epochs = 20, cfg = {}):
    # print("get estimator type: ", model_type)

    if model_type == 'ridge':
        if cfg=={}:
            cfg={'alpha':1.0}
            
        estimator = linear_model.Ridge(alpha = cfg['alpha'])
        
    elif model_type == 'xgb':
        if cfg=={}:
            cfg = {'lr':0.08, 'gamma':0.0, 'subsample':0.75, 'cols_bt':1, 'maxdepth':7}

        # print("call xgb estimator with cfg:", cfg)    
        estimator = xgb.XGBRegressor(n_estimators=cfg['n_estimators'], 
                                     learning_rate = cfg['lr'], 
                                     gamma = cfg['gamma'], 
                                     subsample = cfg['subsample'], 
                                     colsample_bytree = cfg['cols_bt'], 
                                     max_depth = cfg['maxdepth'])
        
    elif model_type == 'mlp':     
        t.mlp_cfg['lr'] = cfg['lr']   # pass parameters to fn "simple_mlp" via global variable t.mlp_cfg
        # t.mlp_cfg['input_dim'] = X.shape[1]
        estimator = KerasRegressor(build_fn=mlp, epochs=epochs, 
                                   batch_size=cfg['batch_size'], verbose=0)
        
    elif model_type == 'lstm':
        # t.mlp_cfg['lr'] = cfg['lr']   # pass parameters to fn "simple_mlp" via global variable t.mlp_cfg
        # t.mlp_cfg['input_dim'] = X.shape[1]
        estimator = KerasRegressor(build_fn=lstm, epochs=epochs, 
                                   batch_size=cfg['batch_size'], verbose=1)
        
    elif model_type == 'multi_lstm':
        # t.mlp_cfg['lr'] = cfg['lr']   # pass parameters to fn "simple_mlp" via global variable t.mlp_cfg
        # t.mlp_cfg['input_dim'] = X.shape[1]
        estimator = KerasRegressor(build_fn=multi_lstm, epochs=epochs, 
                                   batch_size=cfg['batch_size'], verbose=1)
              
    else:
        print("no valid estimator type: ", model_type)
        estimator = None
        
    return estimator


# create and train model and evaluate by cross validation
def eval_cv(model_type, X, Y, cfg = {}, epochs=0, splits = 3):
    
    # metrics = loss(self.monitor, ','.join(list(logs.keys())))
    # metrics = logs.keys()
    
    # print(metrics)
    
    fit_params = {}
    
    if model_type == 'mlp':
        callbacks = []
        if t.mlp_cfg['earlystop']:
            callbacks.append(EarlyStopping(monitor='loss', min_delta=0.0001, patience=100, verbose=1, mode='auto'))
            # fit_params = {'callbacks': [EarlyStopping(monitor='loss', min_delta=0.0001, patience=100, verbose=1, mode='auto')]}
            print("evaluating with early stopping")
            
        if t.mlp_cfg['lr_exp_decay']:
            t.mlp_cfg['lr'] = cfg['lr']   # pass parameters to fn "simple_mlp" via global variable t.mlp_cfg
            t.mlp_cfg['k_exp'] = cfg['k_exp']   
            lrate = LearningRateScheduler(exp_decay)
            callbacks.append(lrate)
            print("evaluating with exponential decay")

        fit_params = {'callbacks': callbacks}
    
    estimator = get_estimator(model_type, X, epochs, cfg)

    kfold = KFold(n_splits=splits, random_state=t.seed)
    # enforce MSE as scoring, to get comparable results for different models /  cross_validate() more flexible
    print("call cross_val_score {} epochs, with config {}".format(epochs,cfg))
    #print("X.shape",X.shape)
    #print("X",X[0])    
    results = cross_val_score(estimator, X, Y, cv=kfold, scoring='neg_mean_squared_error', verbose=1,
                              fit_params=fit_params)
    print("MSE {}: mean *** {:.5f} *** std: {:.4f}".format(model_type, -results.mean(), results.std()))
    # print("MSE of all Folds: {}".format(np.round(-results,5)))
    return results

