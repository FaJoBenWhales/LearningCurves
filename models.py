# Deep learning Theory final project
import numpy as np
import time
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

# stores weights of best values and restore these after training --> not done by default by earlystop !! 
# modified keras ModelCheckpoint class see https://github.com/keras-team/keras/issues/2768 code user louis925
# enhanced with "reset" parameter
import warnings
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
                print("choose max as mode")
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                print("choose min as mode")                
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


# mode "nextstep": train on predicting last point of observed points (e.g. 5 of 5)
# mode "finalstep": train on predicting final point of learning curve (#40)
def _train_lstm(model, X, steps, idx, batch_size, epochs, callbacks = None, mode = "nextstep"):

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

    # generator for sequences for fit_generator()
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
        print("train considering", steps[0], "epochs, evaluate with", steps[1], "epochs")
    else:
        print("train with random nr. of epochs, evaluate with", steps[1], "epochs")
        
    train_idx, val_idx = idx[0], idx[1]
    train_steps, val_steps = steps[0], steps[1]

    train_generator = generate_seqs(X, train_steps, train_idx, batch_size, mode)        
    val_generator = generate_seqs(X, val_steps, val_idx, batch_size, mode)        
        
    hist = model.fit_generator(train_generator,
                               steps_per_epoch = int(np.ceil(train_idx.shape[0] / batch_size)), 
                               epochs=epochs, 
                               callbacks=callbacks,
                               validation_data = val_generator,
                               validation_steps = int(np.ceil(val_idx.shape[0] / batch_size)), 
                               verbose = 0)
    return hist


# taking train/valid split point as parameter for manual experiments
# steps tuple (training-timesteps , validation-timesteps), train-steps == 0 --> random length
def train_lstm(model, X, steps=(10,10), split=200, batch_size=20, epochs=3, mode = 'nextstep'):
    
    if mode == "nextstep" or mode == "finalstep":
        sample_no = (X[0] if type(X) == list else X).shape[0]
        idx = (np.arange(0,split), np.arange(split,sample_no))
        return _train_lstm(model, X, steps, idx, batch_size, epochs, callbacks = None, mode=mode)
    else: 
        print("unknown mode", mode)
        
# evaluation on next (mode='nextstep') or on final (mode='finalstep') step
def _eval_lstm(model, X, steps, idx, batch_size, mode = 'nextstep'):

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

    score, mse = model.evaluate(X_val, Y_val, batch_size=batch_size) 
                          
    print("mse: ", mse)  
    return mse    
        
        
def eval_lstm(model, X, steps, split, batch_size, mode = 'nextstep'):

    sample_no = (X[0] if type(X) == list else X).shape[0]
    idx = np.arange(split,sample_no)               # taking all data after split for valuation
    
    return(_eval_lstm(model, X, steps, idx, batch_size, mode = 'nextstep'))    


def _pred_finalpoints(model, X, steps, idx, batch_size=20):

    configs, lcs = X[0], X[1]
    lcs_trunc = lcs[idx][:,:steps]

    # extrapolate series step by step based on observed curve (lcs_trunc, starting at "steps")
    while lcs_trunc.shape[1] < 40:
        # compute next step, and append prediction to series of observed points
        nextsteps = model.predict([configs[idx], lcs_trunc], batch_size=batch_size)
        nextsteps = nextsteps.reshape((lcs_trunc.shape[0], 1, 1))
        lcs_trunc = np.append(lcs_trunc,nextsteps,axis=1)

    # compare extrapolated final point with true value
    y_pred = lcs_trunc[:,-1].reshape(idx.shape[0])
    y_true = lcs[idx][:,-1].reshape(idx.shape[0])
    mse = ((y_pred - y_true) ** 2).mean()

    return mse



# function assumes X is tuple [configs, lcs] - function for external call via "split" value
def pred_finalpoints(model, X, steps, split=200, batch_size=20):

    sample_no = X[0].shape[0]
    idx_train = np.arange(0,split)
    idx_val   = np.arange(split,sample_no)
    
    mse_train = _pred_finalpoints(model, X, steps, idx_train, batch_size=batch_size)
    mse_val =   _pred_finalpoints(model, X, steps, idx_val,   batch_size=batch_size)
                          
    print("mse train: {:.5f}, mse validation {:.5f}".format(mse_train, mse_val))  
    
    # return mse_train, mse_val


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
# steps = (training steps, lsit of validation steps) e.g. (10,[5,10,20,30])
def eval_cv(model_type, X, Y, steps=(5,[5]), cfg={}, epochs=0, splits=3, 
            lr_exp_decay=False, earlystop=False, dropout=False, L1L2=False, 
            mode='nextstep'):
    
    print("cross validate {} epochs, train on {} steps, validate on {} steps".format(epochs, steps[0], steps[1]))
    print("config {}".format(cfg))

    callbacks = []
    if earlystop==True:
        monitor = 'val_loss' if model_type == 'lstm' or model_type == 'multi_lstm' else 'loss'
         #callbacks.append(EarlyStopping(monitor='loss', min_delta=0.0001, 
        callbacks.append(EarlyStopping(monitor=monitor, min_delta=0.0001,                                        
                                       patience=np.amin([np.amax([epochs/10, 5]),75]), 
                                       verbose=1, mode='auto'))
        # save weights at best iteration, and restore at end of training
        # takes best iteration of all CV-folds
        callbacks.append(GetBest(monitor=monitor, verbose=1, mode='auto', reset = True))
        print("evaluating with early stopping")

    if lr_exp_decay==True:
        lrate = MyLearningRateScheduler(exp_decay, init_lr = cfg['lr'], k_exp=cfg['k_exp'])
        callbacks.append(lrate)
        print("evaluating with exponential decay")

    kfold = KFold(n_splits=splits, random_state=t.seed)

    # work around with own cross validation, as cross_val_score does not accept multiple inputs   
    results_val=[]      # generally work only on validation data...
    results_train=[]    # ... but for task with 'nextstep' also on training data

    if model_type == 'lstm' or model_type == 'multi_lstm':  # own implementation of cross validation
    
        model = lstm() if model_type == 'lstm' else multi_lstm()
        Wsave = model.get_weights()
        
        fold_count = 0
        for train_idx, val_idx in kfold.split(Y):
            fold_count += 1
            # faster solution: train once, evaluate over all cases [5,10,20,30]
            model.set_weights(Wsave)     # to make results reproducible always start with same init weights
            print("train fold {} on {} steps, validation on {} steps".format(fold_count, steps[0], steps[0]))
            # steps[1] for validation here only relevant as criterion for early stopping
            hist = _train_lstm(model, X, steps=(steps[0], steps[0]), 
                               idx=(train_idx, val_idx), 
                               batch_size=cfg['batch_size'], 
                               epochs=epochs, 
                               callbacks=callbacks,
                               mode=mode)
            
            # now model has weights of best run during last training (based on val_loss)
            # now evaluate on train and valid data of given steps [5,10,20,30]
            train_mses, val_mses = [], [] 
            for val_steps in steps[1]:

                if mode == 'finalstep':
                    val_mses.append(_eval_lstm(model, X, val_steps, val_idx, cfg['batch_size'], mode = 'finalstep'))
                    train_mses.append(_eval_lstm(model, X, val_steps, train_idx, cfg['batch_size'], mode = 'finalstep'))                  
                elif mode == 'nextstep':
                    val_mses.append(_pred_finalpoints(model, X, val_steps, val_idx, batch_size=cfg['batch_size']))
                    train_mses.append(_pred_finalpoints(model, X, val_steps, train_idx, batch_size=cfg['batch_size']))
                else:
                    print("invalid mode", mode)
                    
                print("validate on {} steps, mse on train / validation data: {:.5f} / {:.5f}"\
                      .format(val_steps, train_mses[-1], val_mses[-1]))

                
            results_val.append(val_mses)
            results_train.append(train_mses)                        

    else:     # for mlp, xgb, ridge etc. use skikitlearn cross_val_score()

        fit_params = {}        
        if model_type == 'mlp':
            fit_params = {'callbacks': callbacks}

        estimator = get_estimator(model_type, X, epochs, cfg, dropout=dropout, L1L2=L1L2)

        # enforce MSE as scoring, to get comparable results for different models 
        results_val = cross_val_score(estimator, X, Y, cv=kfold, scoring='neg_mean_squared_error', verbose=1,
                                  fit_params=fit_params)
        
    results_val, results_train = np.array(results_val), np.array(results_train)
    print("MSE on validation data on {} steps: means over folds: *** {} ***"\
          .format(steps[1], abs(np.round(results_val.mean(axis=0),5))))
    print("Results validation data of all Folds: \n{}".format(np.round(results_val,5)))
    
    if results_train.shape[0] > 0:
        print("MSE on training data on {} steps: means over folds: *** {} ***"\
              .format(steps[1], abs(np.round(results_train.mean(axis=0),5))))
        print("Results training data of all Folds: \n{}".format(np.round(results_train,5)))
    
    return results_train.mean(axis=0), results_val.mean(axis=0) 