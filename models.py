# Deep learning Theory final project
import numpy as np
import time
import keras.backend as K
from keras import optimizers
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, LSTM, concatenate
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping, LearningRateScheduler, Callback
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb

import tools as t

# passing a dictionary (cfg) to mlp triggers a deprecation warning - passing simple parameters does not (??)
def mlp(lr=None, dropout=False, L1L2=False):
    
    # print("create mlp using learning rate:", lr)
    
    if L1L2==True:
        kernel_regularizer=l1_l2(l1=0.01, l2=0.01)
        print("create mlp using L1L2 regularisation")
    else: 
        kernel_regularizer = None

    model = Sequential()
    model.add(Dense(64, input_dim=5, kernel_initializer='normal', 
                    kernel_regularizer=kernel_regularizer, activation='relu'))
    if dropout==True:
        model.add(Dropout(0.2))
        print("create mlp using Dropout")
    model.add(Dense(64, kernel_initializer='normal', 
                    kernel_regularizer=kernel_regularizer, activation='relu'))
    if dropout==True:
        model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='normal'))
    
    opt = optimizers.SGD(lr=lr)
    
    model.compile(loss='mean_squared_error', optimizer=opt)  # no adaptive learning rate (--> later compare to exponential)
    
    return model


def lstm(input_dim=1):
    # usually 1, but changed to 1+5=6 when loading data by load_lstm_data_concat_cfg() 
    # --> adding configs to each data point of learning curves
    print("build lstm with input_dim:", input_dim)
    data_dim = input_dim      # features of input data
    
    output_dim = 1
    
    model = Sequential()
    model.add(LSTM(64, return_sequences=True,
                   # input shape = (timesteps, data_dim) - but timesteps can be None for flexible use
                   input_shape=(None, data_dim)))  # returns a sequence of vectors of dimension 32
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
    
    data_dim = 1
    output_dim = 1
    
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


# ToDo: pass learning rate at creation of model
def train_mlp(model, configs, Y, cfg, split, epochs):
    
    configs_train, configs_val = configs[:split], configs[split:]
    y_train, y_val = Y[:split], Y[split:]
    # print(configs_train.shape, configs_val.shape, y_train.shape, y_val.shape)    

    hist = model.fit(configs_train, y_train,
                     batch_size=cfg['batch_size'], 
                     epochs=epochs,
                     validation_data=(configs_val, y_val))
    return hist


def eval_mlp(model, configs, Y, split, batch_size):
    
    configs_train, configs_val = configs[:split], configs[split:]
    y_train, y_val = Y[:split], Y[split:]             
    
    mse = model.evaluate(configs_val, y_val, batch_size=batch_size)
    print("mse: ", mse)
    
    return mse        


def train_lstm(model, lcs, Y, split, batch_size, epochs):
   
    x_train, x_val = lcs[:split], lcs[split:]
    y_train, y_val = Y[:split], Y[split:]

    hist = model.fit(x_train, y_train,
                     batch_size=batch_size, epochs=epochs,
                     validation_data=(x_val, y_val))
    return hist

    

def eval_lstm(model, lcs, Y, split, batch_size):
    
    x_train, x_val = lcs[:split], lcs[split:]
    y_train, y_val = Y[:split], Y[split:]              
    
    score, mse = model.evaluate(x_val, y_val, batch_size=batch_size)
    print("mse: ", mse)
    
    return mse    
    
    
def train_multi_lstm(model, configs, lcs, Y, split, batch_size, epochs):
    # lcs, Y = t.shape_lstm_data(lcs, Y, timesteps)
    
    x_train, x_val = lcs[:split], lcs[split:]
    y_train, y_val = Y[:split], Y[split:]
    configs_train, configs_val = configs[:split], configs[split:]
    # print(x_train.shape, x_val.shape, configs_train.shape, configs_val.shape, y_train.shape, y_val.shape)    

    hist = model.fit([configs_train, x_train], y_train,
                     batch_size=batch_size, epochs=epochs,
                     validation_data=([configs_val, x_val], y_val))
    return hist


def eval_multi_lstm(model, configs, lcs, Y, split, batch_size):
    x_train, x_val = lcs[:split], lcs[split:]
    y_train, y_val = Y[:split], Y[split:]
    configs_train, configs_val = configs[:split], configs[split:]    
   
    score, mse = model.evaluate([configs_val, x_val], y_val,
                                batch_size=5)
    
    return mse


def exp_decay(epoch, lr, k_exp=1):
    
    initial_lrate = lr    
    k = k_exp  
    lrate = initial_lrate * np.exp(-k*epoch)
    if (epoch % 100 == 0):
        print("new lr: ", lrate)
    return lrate


# copied from https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L591 
# and enhanced in order to facilitate passing parameter k_exp to schedule function exp_decay
class MyLearningRateScheduler(Callback):

    def __init__(self, schedule, init_lr, k_exp=0, verbose=0):
        super(MyLearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.init_lr = init_lr        
        self.k_exp = k_exp     
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(K.get_value(self.model.optimizer.lr))
        lr = self.schedule(epoch, lr=self.init_lr, k_exp=self.k_exp)

        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler reducing learning rate to %s.' % (epoch + 1, lr))


def get_estimator(model_type, X = None, epochs = 20, cfg = {}, **kwargs):

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
        estimator = KerasRegressor(build_fn=mlp, lr = cfg['lr'], 
                                   dropout=kwargs.get('dropout',False), L1L2=kwargs.get('L1L2',False),
                                   epochs=epochs, batch_size=cfg['batch_size'], verbose=0)
        
    elif model_type == 'lstm':
        estimator = KerasRegressor(build_fn=lstm, input_dim = X[0][2].shape[0], 
                                   epochs=epochs, batch_size=cfg['batch_size'], verbose=1)
        
    elif model_type == 'multi_lstm':
        estimator = KerasRegressor(build_fn=multi_lstm, epochs=epochs, 
                                   batch_size=cfg['batch_size'], verbose=1)
              
    else:
        print("no valid estimator type: ", model_type)
        estimator = None
        
    return estimator


# create and train model and evaluate by cross validation
def eval_cv(model_type, X, Y, cfg, epochs=0, splits = 3, 
            lr_exp_decay=False, earlystop=False, dropout=False, L1L2=False):
    
    print("cross validate {} epochs, with config {}".format(epochs,cfg))

    callbacks = []
    if earlystop==True:
        callbacks.append(EarlyStopping(monitor='loss', min_delta=0.0001, patience=100, verbose=1, mode='auto'))
        print("evaluating with early stopping")

    if lr_exp_decay==True:
        lrate = MyLearningRateScheduler(exp_decay, init_lr = cfg['lr'], k_exp=cfg['k_exp'])
        callbacks.append(lrate)
        print("evaluating with exponential decay")

    kfold = KFold(n_splits=splits, random_state=t.seed)        

    # work-around with own cross validation, as cross_val_score does not accept multiple inputs        
    if model_type == 'multi_lstm':
        
        model = multi_lstm()
        results=[]
        
        # in case of multi_lstm X is a tuple [configs, lcs]
        for train, val in k_fold.split(X[0]):
            
            hist = model.fit([X[0][train], X[1][train]], Y[train],
                             batch_size=cfg['batch_size'], 
                             epochs=epochs, 
                             callbacks=callbacks,
                             validation_data=([X[0][val], X[1][val]], Y[val]))
            
            results.append(hist.history['val_mean_squared_error'][-1])
            
        results = np.array(results)
        
    else:

        fit_params = {}        
        if model_type == 'mlp':
            fit_params = {'callbacks': callbacks}

        estimator = get_estimator(model_type, X, epochs, cfg, dropout=dropout, L1L2=L1L2)

        # enforce MSE as scoring, to get comparable results for different models 
        results = cross_val_score(estimator, X, Y, cv=kfold, scoring='neg_mean_squared_error', verbose=1,
                                  fit_params=fit_params)
        
    print("MSE {}: mean *** {:.5f} *** std: {:.4f}".format(model_type, abs(results.mean()), results.std()))
    print("Result of all Folds: {}".format(np.round(results,4)))
    
    return results