# Deep learning Theory final project
import numpy as np
from copy import deepcopy
import time
import warnings
import keras.backend as K
from keras import optimizers
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, LSTM, concatenate, Masking
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping, LearningRateScheduler, Callback
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb

import tools as t


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

            
# stores weights of best values and restore these after training --> not done by default by earlystop !! 
# modified keras ModelCheckpoint class see https://github.com/keras-team/keras/issues/2768 code user louis925
# enhanced with "reset" parameter
class GetBest(Callback):
    """Get the best model at the end of training.
    # Arguments
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        mode: one of {auto, min, max}.
            The decision
            to overwrite the current stored weights is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
    # Example
        callbacks = [GetBest(monitor='val_acc', verbose=1, mode='max')]
        mode.fit(X, y, validation_data=(X_eval, Y_eval),
                 callbacks=callbacks)
    """
    # reset best found value at beginning of each new training
    def __init__(self, monitor='val_loss', verbose=0,
                 mode='auto', period=1, reset = True):
        super(GetBest, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.period = period
        self.reset = reset
        self.best_epochs = 0
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('GetBest mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                # print("choose max as mode")
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                # print("choose min as mode")                
                self.monitor_op = np.less
                self.best = np.Inf
                
    def on_train_begin(self, logs=None):
        if self.reset == True:      # useful if multiple calls of fit(), e.g. during cross validation 
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            #filepath = self.filepath.format(epoch=epoch + 1, **logs)
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can pick best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                              ' storing weights.'
                              % (epoch + 1, self.monitor, self.best,
                                 current))
                    self.best = current
                    self.best_epochs = epoch + 1
                    self.best_weights = self.model.get_weights()
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s is %0.5f, did not improve' %
                              (epoch + 1, self.monitor, current))            
                    
    def on_train_end(self, logs=None):
        if self.verbose > 0:
            print('Using epoch %05d with %s: %0.5f' % (self.best_epochs, self.monitor,
                                                       self.best))
        self.model.set_weights(self.best_weights)


# create model for Baseline 3.4 
def xgb_model(cfg):
    
    model = xgb.XGBRegressor(n_estimators=cfg['n_estimators'], 
                                 learning_rate = cfg['lr'], 
                                 gamma = cfg['gamma'], 
                                 subsample = cfg['subsample'], 
                                 colsample_bytree = cfg['cols_bt'], 
                                 max_depth = cfg['maxdepth'])
    return model
      

# passing a dictionary (cfg) to mlp triggers a deprecation warning - passing simple parameters does not (??)
def mlp(cfg, dropout=False, L1L2=False):
    
    # print("create mlp using learning rate:", lr)
    
    if L1L2==True:
        kernel_regularizer=l1_l2(l1=cfg['l1'], l2=cfg['l2'])
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
    
    opt = optimizers.SGD(lr=cfg['lr'])
    
    model.compile(loss='mean_squared_error', optimizer=opt)  # no adaptive learning rate (--> later compare to exponential)
    
    return model


def lstm(input_dim=1):
    # usually 1, but changed to 1+5=6 when loading data by load_lstm_data_concat_cfg() 
    # --> adding configs to each data point of learning curves
    print("build lstm with input_dim:", input_dim)
    data_dim = input_dim      # features of input data
    
    model = Sequential()
    # masking triggers LSTM to skip all timesteps with value = mask_value (for variable length input)
    model.add(Masking(mask_value=0., input_shape=(None, 1)))    # (timesteps / features)
    model.add(LSTM(64, return_sequences=True,
                   # input shape = (steps, data_dim) - but steps can be None for flexible use
                   input_shape=(None, data_dim)))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(64))  # return a single vector of dimension 32
    model.add(Dense(64))
    model.add(Dense(64))
    model.add(Dense(1))    

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mse'])
    return model


def multi_lstm(lr=0.01):   # keras default is lr=0.001, but runs better at 0.01
    
    # lstm branch of model, using masking for random values
    lcs_input = Input(shape=(None, 1), name='lcs_input')
    # masking = Masking(mask_value=0., input_shape=(None, 1))(lcs_input)    # input (timesteps / features)
    masking = Masking(mask_value=0.)(lcs_input)    # input (timesteps / features)

    lstm_out = LSTM(64, return_sequences=True)(masking)
    lstm_out = LSTM(64)(lstm_out)

    # branch for non sequential configuration data
    cfgs_input = Input(shape=(5,), name='cfgs_input')
    
    x = concatenate([cfgs_input, lstm_out])    
    x = Dense(64)(x)
    x = Dense(64)(x)
    main_output = Dense(1, name='main_output')(x)
    
    model = Model(inputs=[cfgs_input, lcs_input], outputs=[main_output])

    adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mse',
                  optimizer=adam,
                  metrics=['mse'])
    return model


# train xgb to predict next step based on config and last 4 steps
def _train_xgb_next(model, X, idx):
    
    configs, lcs = X[0][idx], X[1][idx]
    lcs = lcs.reshape((idx.shape[0], lcs.shape[1]))   # from (200,1,1) to (200,1)
    
    for lag in np.arange(0,36):
        # print("lcs.shape[1]", lcs.shape[1])
        lcs_next_4 = lcs[:,lag:lag+4]
        print("train on new epoch",lag+5, "true value for curve no. 13 (example)", lcs[13,lag+4])
        # lcs_next_4 = lcs_next_4.reshape(idx.shape[0],4)   # from (200,1,1) to (200,1)
        # for each sample append slice of next for values to config array as input for xgb
        # print("lcs_next_4",len(lcs_next_4))
        # print("configs",len(configs))

        next_input = np.append(configs,lcs_next_4,axis=1)
        next_label = lcs[:,lag+4]
        # print("next_input", next_input[13])
        # print("next_label", next_label[13])
        model.fit(next_input, next_label)

# taking train/valid split point as parameter for manual experiments
# steps tuple (training-timesteps , validation-timesteps), train-steps == 0 --> random length
def train_xgb_next(model, X, split=200):
    
    idx = np.arange(0,split)
    return _train_xgb_next(model, X, idx)
        
        
# predict final point via successive / stepwise prediction based on config + 4 last steps
def _pred_xgb_stepwise(model, X, val_steps, idx):

    print("\neval_xgb starting at step", val_steps)
    configs = X[0][idx] 
    lcs     = X[1][idx]
    lcs     = lcs.reshape((idx.shape[0], lcs.shape[1]))   # from (200,40,1) to (200,40)
    lcs_new = deepcopy(lcs)    # will be manipulated during prediction
    print("lcs", lcs[13])
    
    # go through learning-curves, replacing values stepwise with predictions based on last 4-slice
    for lag in np.arange(val_steps-4,36):
        lcs_next_4 = lcs_new[:,lag:lag+4]
        # for each sample append slice of next for values to config array as input for xgb
        next_input = np.append(configs,lcs_next_4,axis=1)
        # print("next_input", next_input[13])        
        next_pred = model.predict(next_input)
        # print("next_pred", next_pred[13])
        lcs_new[:,lag+4] = next_pred   # set next step in Learning curve with prediction
        # print("lc number 7 with new prediction ", next_pred[7], "\n", lcs_new[7][:lag+5])
        print("step nr.", lag+4 ,"prediction / true value for lc number 13", next_pred[13], "/", lcs[13][lag+4])
    
    # compare extrapolated final points with true values
    y_pred = lcs_new[:,-1].reshape(idx.shape[0])
    # y_true = lcs[:,-1].reshape(idx.shape[0])
    
    # mse = ((y_pred - y_true) ** 2).mean()
    
    return y_pred

# function assumes X is tuple [configs, lcs] - function for external call via "split" value
def eval_xgb_stepwise(model, X, Y, steps, split=200):

    sample_no = (X[0] if type(X) == list else X).shape[0]
    idx_trn = np.arange(0,split)
    idx_val = np.arange(split,sample_no) 
    
    y_pred_trn = _pred_xgb_stepwise(model, X, steps, idx_trn)
    y_pred_val = _pred_xgb_stepwise(model, X, steps, idx_val)
    
    mse_trn = ((y_pred_trn - Y[idx_trn]) ** 2).mean()
    mse_val = ((y_pred_val - Y[idx_val]) ** 2).mean()
    print("mse train: {:.5f}, mse validation {:.5f}".format(mse_trn, mse_val))  
    
    return mse_trn, mse_val

# ToDo: pass learning rate at creation of model
def train_mlp(model, configs, Y, cfg, split, epochs):
    
    configs_trn, configs_val = configs[:split], configs[split:]
    y_trn, y_val = Y[:split], Y[split:]
    # print(configs_trn.shape, configs_val.shape, y_trn.shape, y_val.shape)    

    hist = model.fit(configs_trn, y_trn,
                     batch_size=cfg['batch_size'], 
                     epochs=epochs,
                     validation_data=(configs_val, y_val))
    return hist

def _pred_mlp(model, X, idx, batch_size):

    configs = X[idx]
    y_pred = model.predict(configs, batch_size=batch_size)
    y_pred = y_pred.reshape(idx.shape[0])    
    
    return y_pred

def eval_mlp(model, X, Y, split, batch_size):
    
    sample_no = (X[0] if type(X) == list else X).shape[0]
    idx_trn = np.arange(0,split)
    idx_val = np.arange(split,sample_no)     
    
    y_pred_trn = _pred_mlp(model, X, idx_trn, batch_size)    
    y_pred_val = _pred_mlp(model, X, idx_val, batch_size)

    mse_trn = ((y_pred_trn - Y[idx_trn]) ** 2).mean()
    mse_val = ((y_pred_val - Y[idx_val]) ** 2).mean()
    print("mse train: {:.5f}, mse validation {:.5f}".format(mse_trn, mse_val))     
    
    return mse_trn, mse_val    


# mode "nextstep": train on predicting last point of observed points (e.g. 5 of 5)
# mode "finalstep": train on predicting final point of learning curve (#40)
def _train_lstm(model, X, steps, idx, batch_size, epochs, callbacks = None, mode = "nextstep", verbose = 0):
    
    # cut learning curves at "steps", return x and y values to pass to model
    def truncate_lcs(lcs, steps, mode):
        if steps != 0:                         # truncate lenght of training sequences to given value     
            lcs_trunc = lcs[:,:steps]      # whole observed sequence as training (up to step=5 or so)   
            if mode == "nextstep":  
                y = lcs[:,steps]               # next point after observed sequence is target
            elif mode == "finalstep":
                y = lcs[:,-1]                  # final point (40) of total seqquence is target
            # print("lcs_trunc", lcs_trunc[3:7])
            # print("lcs_trunc + 2", lcs[3:7,:steps+2])
            # print("y", y[3:7])
        else:                                         # truncate train seqs randomly by masking
            lcs_trunc = lcs
            y = np.zeros((lcs.shape[0],1))
            seq_lens = np.random.randint(low=5, high=20, size=lcs.shape[0])
            for i in range(lcs.shape[0]):
                y[i] = lcs[i][-1] if mode == "finalstep" else lcs[i][seq_lens[i]]
                # print("y[i]", y[i])                 
                lcs_trunc[i][seq_lens[i]:] = 0      # mask all values after end of seqeunce with 0  
                # print("lcs_trunc[i]", lcs_trunc[i]) 
                # print("y[i] after", y[i]) 
                
        return lcs_trunc, y

    # generate input for fit_generator(), with chosen step-length
    def generate_seqs(X, steps, idx, batch_size, mode):

        while 1:

            if not type(X) is list:                         # X is not tuple --> simple lstm without configs    
                lcs = X[idx]                                # select samples according to index array
                x,y = truncate_lcs(lcs, steps, mode)  # 
                for i in range(0, y.shape[0], batch_size):
                    yield (x[i : i+batch_size], y[i : i+batch_size])              
            else:                                           # X is tuple (configs,lcs)                
                configs = X[0][idx]
                lcs     = X[1][idx]
                x, y = truncate_lcs(lcs, steps, mode)

                for i in range(0, y.shape[0], batch_size):
                    # print("yield x, y", x[7], y[7])
                    yield ([configs[i : i+batch_size], x[i : i+batch_size]], y[i : i+batch_size])      
                    
    if steps[0]!=0:
        print("train on", mode, "considering", steps[0], "epochs, eval during training with", steps[1], "epochs")
    else:
        print("train on", mode, "with random nr. of epochs, eval during training with", steps[1], "epochs")
        
    trn_idx, val_idx = idx[0], idx[1]
    trn_steps, val_steps = steps[0], steps[1]

    trn_generator = generate_seqs(X, trn_steps, trn_idx, batch_size, mode)        
    val_generator = generate_seqs(X, val_steps, val_idx, batch_size, mode)        
        
    hist = model.fit_generator(trn_generator,
                               steps_per_epoch = int(np.ceil(trn_idx.shape[0] / batch_size)), 
                               epochs=epochs, 
                               callbacks=callbacks,
                               validation_data = val_generator,
                               validation_steps = int(np.ceil(val_idx.shape[0] / batch_size)), 
                               verbose = verbose)
    return hist


# taking train/valid split point as parameter for manual experiments
# steps tuple (training-timesteps , validation-timesteps), train-steps == 0 --> random length
def train_lstm(model, X, steps=(10,10), split=200, batch_size=20, epochs=3, mode = 'nextstep', verbose = 0):
    
    if mode == "nextstep" or mode == "finalstep":
        sample_no = (X[0] if type(X) == list else X).shape[0]
        idx = (np.arange(0,split), np.arange(split,sample_no))
        return _train_lstm(model, X, steps, idx, batch_size, epochs, 
                           callbacks = None, mode=mode, verbose=verbose)
    else: 
        print("unknown mode", mode)
        
# evaluation on next (mode='nextstep') or on final (mode='finalstep') step
def _pred_lstm_direct(model, X, steps, idx, batch_size, mode = 'finalstep'):

    if not type(X) is list:                         # simple lstm without configs
        print("evaluate lstm without consideration of configs")
        X_val = X[idx][:,:steps] 
        Y_val = X[idx][:,-1] if mode == 'finalstep' else X[idx][:,steps] 
        # Y_val = X[idx][:,-1]
    else:                                           # X is tuple (configs,lcs)
        print("evaluate lstm with consideration of configs")
        configs, lcs = X[0], X[1]
        lcs_val   = lcs[idx][:,:steps]
        X_val     = [configs[idx], lcs_val]     
        Y_val     = lcs[idx][:,-1] if mode == 'finalstep' else lcs[idx][:,steps]  

    y_pred = model.predict(X_val, batch_size=batch_size) 
    y_pred = y_pred.reshape(idx.shape[0])
    
    return y_pred

# predict next or final step        
def eval_lstm_direct(model, X, Y, steps, split, batch_size):

    sample_no = (X[0] if type(X) == list else X).shape[0]
    idx_trn = np.arange(0,split)
    idx_val = np.arange(split,sample_no)    

    y_pred_trn = _pred_lstm_direct(model, X, steps, idx_trn, batch_size, mode = 'finalstep')
    y_pred_val = _pred_lstm_direct(model, X, steps, idx_val, batch_size, mode = 'finalstep')
    
    mse_trn = ((y_pred_trn - Y[idx_trn]) ** 2).mean()
    mse_val = ((y_pred_val - Y[idx_val]) ** 2).mean()
    print("mse train: {:.5f}, mse validation {:.5f}".format(mse_trn, mse_val))  
    
    return mse_trn, mse_val


# predict final point via stepwise predictions 
def _pred_lstm_stepwise(model, X, steps, idx, batch_size=20):

    configs, lcs = X[0][idx], X[1][idx]
    lcs_trunc = lcs[:,:steps]

    # extrapolate series step by step based on observed curve (lcs_trunc, starting at "steps")
    while lcs_trunc.shape[1] < 40:
        # compute next step, and append prediction to series of observed points
        nextsteps = model.predict([configs, lcs_trunc], batch_size=batch_size)
        nextsteps = nextsteps.reshape((lcs_trunc.shape[0], 1, 1))
        lcs_trunc = np.append(lcs_trunc,nextsteps,axis=1)

    # compare extrapolated final point with true value
    y_pred = lcs_trunc[:,-1].reshape(idx.shape[0])
    
    # y_true = lcs[:,-1].reshape(idx.shape[0])
    # mse = ((y_pred - y_true) ** 2).mean()
    # print("mse stepwise internally", mse)
    
    return y_pred
    

    # return mse

# function assumes X is tuple [configs, lcs] - function for external call via "split" value
def eval_lstm_stepwise(model, X, Y, steps, split=200, batch_size=20):

    sample_no = (X[0] if type(X) == list else X).shape[0]
    idx_trn = np.arange(0,split)
    idx_val = np.arange(split,sample_no) 
    
    y_pred_trn = _pred_lstm_stepwise(model, X, steps, idx_trn, batch_size=batch_size)
    y_pred_val = _pred_lstm_stepwise(model, X, steps, idx_val, batch_size=batch_size)
    
    mse_trn = ((y_pred_trn - Y[idx_trn]) ** 2).mean()
    mse_val = ((y_pred_val - Y[idx_val]) ** 2).mean()
    print("mse train: {:.5f}, mse validation {:.5f}".format(mse_trn, mse_val))  
    
    return mse_trn, mse_val



'''
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
        
    # elif model_type == 'mlp':     
    #    estimator = KerasRegressor(build_fn=mlp, lr = cfg['lr'], 
    #                               dropout=kwargs.get('dropout',False), L1L2=kwargs.get('L1L2',False),
    #                               epochs=epochs, batch_size=cfg['batch_size'], verbose=0)
              
    else:
        print("no valid estimator type: ", model_type)
        estimator = None
        
    return estimator
'''

# create and train model and evaluate by cross validation
# steps = (training steps, lsit of validation steps) e.g. (10,[5,10,20,30])
def eval_cv(model_type, X, Y, steps=(0,[0]), cfg={}, epochs=0, splits=3, 
            lr_exp_decay=False, earlystop=False, dropout=False, L1L2=False, 
            mode='nextstep'):
    
    print("cross validate {} epochs, train on {} steps, validate on {} steps".format(epochs, steps[0], steps[1]))
    print("config {}".format(cfg))

    callbacks = []
    if earlystop==True:
        callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.00001,                                        
                                       patience=np.amin([np.amax([epochs/10, 5]),75]), 
                                       verbose=1, mode='auto'))
        # save weights at best iteration, and restore at end of training
        # reset = True --> takes best iteration for each CV-fold
        callbacks.append(GetBest(monitor='val_loss', verbose=1, mode='auto', reset = True))
        print("evaluating with early stopping")

    if lr_exp_decay==True:
        lrate = MyLearningRateScheduler(exp_decay, init_lr = cfg['lr'], k_exp=cfg['k_exp'])
        callbacks.append(lrate)
        print("evaluating with exponential decay")


#     # own implementation of cross validation
#    if model_type in ['lstm','multi_lstm','mlp','xgb_next']:
#    # if model_type == 'lstm' or model_type == 'multi_lstm' or model_type == 'mlp' or model_type == 'xgb_next':  

    if model_type == 'ridge':
        model = linear_model.Ridge(alpha = cfg['alpha'])
    elif model_type in ['xgb_next','xgb']:
        model = xgb_model(cfg)
    elif model_type == 'mlp':
        model = mlp(cfg, dropout=dropout, L1L2=L1L2)
    elif model_type == 'lstm':
        model = lstm()
    elif model_type == 'multi_lstm':
        model = multi_lstm(cfg['lr'])
    else:
        print("invalid model type", model_type)

    if model_type in ['lstm','multi_lstm','mlp']:
        Wsave = model.get_weights()

    results_val=[]    # list of validation results for each fold
    results_trn=[]    # ... but for task with 'nextstep' also on training data
    y_pred = np.zeros(Y.shape[0])  # successively stores predictions on validation folds (on all unseen data)
        
    fold_count = 0
    kfold = KFold(n_splits=splits, random_state=t.seed)
    for trn_idx, val_idx in kfold.split(Y):
        # trn_true, val_true = Y[trn_idx], Y[val_idx]
        fold_count += 1
        # faster solution: train once, evaluate over all cases [5,10,20,30]
        if model_type in ['lstm','multi_lstm','mlp']:            
            model.set_weights(Wsave)     # to make results reproducible always start with same init weights
        print("train fold {} on {} steps, validation on {} steps".format(fold_count, steps[0], steps[0]))
        # steps[1] for validation here only relevant as criterion for early stopping
        if model_type in ['xgb','ridge']:
            model.fit(X[trn_idx], Y[trn_idx])
        elif model_type == 'mlp':
            model.fit(X[trn_idx], Y[trn_idx],
                      batch_size=cfg['batch_size'], epochs=epochs, callbacks=callbacks,
                      verbose = 0, validation_data=(X[val_idx], Y[val_idx]))                
        elif model_type == 'xgb_next':
            _train_xgb_next(model, X,trn_idx)
        elif model_type in ['lstm','multi_lstm']:
            _train_lstm(model, X, steps=(steps[0], steps[0]), idx=(trn_idx, val_idx), 
                        batch_size=cfg['batch_size'], epochs=epochs, 
                        callbacks=callbacks, mode=mode)

        # now model has weights of best run during last training (based on val_loss)
        # now evaluate on train and valid data of given steps [5,10,20,30]
        trn_mses, val_mses = [],[]    # list of mses of folds
        trn_pred, val_pred = [],[]    # list of predictions of one fold
        trn_true = Y[trn_idx].reshape(trn_idx.shape[0])
        val_true = Y[val_idx].reshape(val_idx.shape[0])            
        
        if model_type in ['ridge', 'xgb']:
            trn_pred = model.predict(X[trn_idx]).reshape(trn_idx.shape[0])
            val_pred = model.predict(X[val_idx]).reshape(val_idx.shape[0])
            trn_mses.append(((trn_pred - trn_true) ** 2).mean())
            val_mses.append(((val_pred - val_true) ** 2).mean())                
        elif model_type == 'mlp':
            trn_pred = _pred_mlp(model, X, trn_idx, batch_size=cfg['batch_size'])                
            val_pred = _pred_mlp(model, X, val_idx, batch_size=cfg['batch_size'])
            trn_mses.append(((trn_pred - trn_true) ** 2).mean())
            val_mses.append(((val_pred - val_true) ** 2).mean())                
        else:   # if lstm or xgb_next, list of validation data
            for val_steps in steps[1]:
                if model_type == 'xgb_next':
                    val_pred = _pred_xgb_stepwise(model, X, val_steps, val_idx)
                    trn_pred = _pred_xgb_stepwise(model, X, val_steps, trn_idx)
                else:    # lstm
                    if mode == 'finalstep':
                        val_pred = _pred_lstm_direct(model, X, val_steps, val_idx, 
                                                     cfg['batch_size'], mode = 'finalstep')
                        trn_pred = _pred_lstm_direct(model, X, val_steps, trn_idx,
                                                     cfg['batch_size'], mode = 'finalstep')
                    elif mode == 'nextstep':
                        val_pred = _pred_lstm_stepwise(model, X, val_steps, val_idx,
                                                          batch_size=cfg['batch_size'])
                        trn_pred = _pred_lstm_stepwise(model, X, val_steps, trn_idx,
                                                            batch_size=cfg['batch_size'])
                    else:
                        print("invalid mode", mode)

                trn_mses.append(((trn_pred - trn_true) ** 2).mean())
                val_mses.append(((val_pred - val_true) ** 2).mean())
                # print("Y.shape after", Y.shape)

                print("validate on {} steps, mse on train / validation data: {:.5f} / {:.5f}"\
                      .format(val_steps, trn_mses[-1], val_mses[-1]))

        y_pred[val_idx] = val_pred   # store results only for last value in list of val_steps

        results_val.append(val_mses)
        results_trn.append(trn_mses)
    '''    
    # end for trn_idx, val_idx in kfold.split(Y):
    else:     # for xgb, ridge etc. use skikitlearn cross_val_score()

        fit_params = {}        
        if model_type == 'mlp':
            fit_params = {'callbacks': callbacks}

        estimator = get_estimator(model_type, X, epochs, cfg, dropout=dropout, L1L2=L1L2)

        # enforce MSE as scoring, to get comparable results for different models 
        results_val = cross_val_score(estimator, X, Y, cv=kfold, scoring='neg_mean_squared_error', verbose=1,
                                  fit_params=fit_params)
    '''    
    results_val, results_trn = np.array(results_val), np.array(results_trn)
    val_means, trn_means = [], []
    
    val_means = abs(np.round(results_val.mean(axis=0),5))
    print("MSE on validation data on {} steps: means over folds: *** {} ***".format(steps[1], val_means))
    print("Results validation data of all Folds: \n{}".format(np.round(results_val,5)))
    
    if results_trn.shape[0] > 0:
        trn_means = abs(np.round(results_trn.mean(axis=0),5))
        print("MSE on train data on {} steps: means over folds: *** {} ***".format(steps[1], trn_means))
        print("Results training data of all Folds: \n{}".format(np.round(results_trn,5)))
        
    mse_total = ((y_pred - Y.reshape(Y.shape[0])) ** 2).mean()
    print("mse over all validation data", mse_total)

    result = {'y_pred'    : y_pred,
              'mse'       : mse_total, 
              'trn_means' : trn_means, 
              'val_means' : val_means}
        
    return result    
    # return y_pred, mse_total, trn_means, val_means