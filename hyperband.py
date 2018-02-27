# Deep learning Theory final project
import numpy as np
import os
import os.path
import pickle
import keras
import time

import ConfigSpace as CS
import hpbandster.distributed.utils
from hpbandster.distributed.worker import Worker
import logging
logging.basicConfig(level=logging.ERROR)  # verbosity of hyperband ?

import tools as t
import models as m


def configuration_space_from_raw(hpRaw, hpRawConditions, resolve_multiple='AND'):
    cs = CS.ConfigurationSpace()
    #
    # add hyperparameters
    #
    for hp in hpRaw:
        if hp[4] == "float":
            cs.add_hyperparameter(
                CS.UniformFloatHyperparameter(
                    hp[0],
                    lower=hp[1][0],
                    upper=hp[1][1],
                    default_value=hp[2],
                    log=hp[3]
                )
            )
        elif hp[4] == "int":
            cs.add_hyperparameter(
                CS.UniformIntegerHyperparameter(
                    hp[0],
                    lower=hp[1][0],
                    upper=hp[1][1],
                    default_value=hp[2],
                    log=hp[3]
                )
            )
        elif (hp[4] == "cat"):
            cs.add_hyperparameter(
                CS.CategoricalHyperparameter(
                    hp[0],
                    hp[1]
                )
            )
        else:
            raise Exception("unknown hp type in hpRawList")

    #
    # add conditions
    #
    covered_conditions = dict()
    for cond in hpRawConditions:
        # check if conditions for that hyperparameter were already processed
        if cond[0] in covered_conditions:
            continue
        covered_conditions[cond[0]] = True
        
        # get all conditions for that hyperparameter
        all_conds_for_hyperparameter = []
        for other_cond in hpRawConditions:
            if other_cond[0] == cond[0]:
                all_conds_for_hyperparameter.append(other_cond)
        
        # create the condition objects
        condition_objects = []
        for cond in all_conds_for_hyperparameter:
            if cond[1] == "eq":
                condition_objects.append(
                    CS.EqualsCondition(
                        cs.get_hyperparameter(cond[0]),
                        cs.get_hyperparameter(cond[2]),
                        cond[3]))
            elif cond[1] == "gtr":
                condition_objects.append(
                    CS.GreaterThanCondition(
                        cs.get_hyperparameter(cond[0]),
                        cs.get_hyperparameter(cond[2]),
                        cond[3]))
            else:
                raise Exception("unknown condition type in hpRawConditions")
        
        # add the conditons to the configuration space
        if len(condition_objects) == 1:
            # simply add the condition
            cs.add_condition(condition_objects[0])
        else:
            # resolve multiple conditions
            if resolve_multiple == 'AND':
                cs.add_condition(
                    CS.AndConjunction(*condition_objects))
            elif resolve_multiple == 'OR':
                cs.add_condition(
                    CS.OrConjunction(*condition_objects))
            else:
                raise Exception("resolve_multiple=", resolve_multiple, ". should be 'AND' or 'OR'")
    
    return cs


def get_keras_config_space(model_type):
    
    if model_type == 'ridge':
        hpRaw = [
            ['alpha',                     [0.1, 10],            1.0,        True,   'float'],
        ]
        
    elif model_type == 'xgb':
        hpRaw = [
            #<    name              >   <  Range       >      <Default>     <Log>   <Type>
            ['n_estimators',             [30, 300],          100,           False,  'int'],            
            ['lr',                       [0.001, 1.0],       0.3,           True,   'float'],
            ['gamma',                    [0.000, 1.0],       0.001,         False,   'float'],
            ['subsample',                [0.001, 1.0],       1.0,           False,   'float'],
            ['cols_bt',                  [0.001, 1.0],       1.0,           False,   'float'],
            ['maxdepth',                 [3., 10.],          6.0,           False,   'int'],             
        ]
        
    elif model_type == 'mlp':
        if t.mlp_cfg['lr_exp_decay']:
            hpRaw = [
                ['lr',                       [0.02, 0.2],     0.2,            True,    'float'],
                ['k_exp',                    [0.001,0.1],     0.04,           True,    'float'],
                ['batch_size',               [16, 32],         32,            True,    'int'],            
            ]
        else:
            hpRaw = [
                ['lr',                       [0.0001, 0.5],     0.05,          True,    'float'],
                ['batch_size',               [16, 64],           32,           True,    'int'],
            ] 
 
    elif model_type == 'lstm':
        hpRaw = [
            # ['lr',                       [0.0001, 0.5],     0.05,          True,    'float'],
            ['batch_size',                   [1, 10],           10,           True,    'int'],
        ] 

    else:
        print("invalid model type: ", model_type)
        
    hpRawConditions = [
        #< conditional hp name      >   <cond. Type>    <cond. variable>        <cond. value>
        # ["num_dense_units_1",           "gtr",          "num_dense_layers",     1],
        # ["num_dense_units_2",           "gtr",          "num_dense_layers",     2],
        # ["num_dense_units_3",           "eq",           "num_dense_layers",     4],
    ]
    
    return configuration_space_from_raw(hpRaw, hpRawConditions, resolve_multiple='AND')


def objective_crossval(X, Y, config, epochs, model_type, save_data_path="plots", *args, **kwargs):
    
    start_time = time.time()

    results = m.eval_cv(model_type, X, Y, cfg=config, epochs=epochs, splits = 4) 

    runtime = time.time() - start_time
    histories = [1,2,3]   # dummy for history
    
    return -results.mean(), runtime, histories


class WorkerWrapper(Worker):
    def __init__(self, X, Y, model_type, save_data_path, objective_function, *args, **kwargs):
        self.X = X
        self.Y = Y
        self.model_type = model_type    # 'ridge', 'mlp' etc. 
        self.save_data_path = save_data_path
        self.objective_function = objective_function
        super().__init__(*args, **kwargs)
    
    def compute(self, config, budget, *args, **kwargs):
        try:
            # print("worker compute ", self.model_type)
            loss, runtime, histories = self.objective_function(
                self. X, self.Y, 
                config,
                epochs=int(budget),
                model_type=self.model_type,
                base_path=self.save_data_path,
                *args, **kwargs)
        finally:
            keras.backend.clear_session()  # avoids problems with multithreading
        return {
            'loss': loss,
            'info': {"runtime": runtime,
                     "histories": histories}
        }

    
def optimize(X,Y,
             objective=objective_crossval,
             config_space_getter=get_keras_config_space,
             model_type='ridge',
             min_budget=1,
             max_budget=128,
             job_queue_sizes=(0, 1),
             base_path="plots",
             run_name=""):
    
    run_name = t.get_run_name(prefix="hyperband", additional=run_name)
    path = os.path.join(base_path, run_name)
    if not os.path.isdir(path):
        os.makedirs(path)
    
    nameserver, ns_port = hpbandster.distributed.utils.start_local_nameserver()
    # starting the worker in a separate thread
    w = WorkerWrapper(X, Y, 
                      model_type=model_type,
                      save_data_path=path,
                      objective_function=objective,
                      nameserver=nameserver,
                      ns_port=ns_port)
    w.run(background=True)

    cs = config_space_getter(model_type)
    configuration_generator = hpbandster.config_generators.RandomSampling(cs)

    # instantiating Hyperband with some minimal configuration
    HB = hpbandster.HB_master.HpBandSter(
        config_generator=configuration_generator,
        run_id='0',
        eta=2,  # defines downsampling rate
        min_budget=min_budget,  # minimum number of epochs / minimum budget
        max_budget=max_budget,  # maximum number of epochs / maximum budget
        nameserver=nameserver,
        ns_port=ns_port,
        job_queue_sizes=job_queue_sizes,
    )
    # runs one iteration if at least one worker is available
    n_iters = int(np.log2(max_budget / min_budget))
    res = HB.run(n_iters, min_n_workers=1)

    # shutdown the worker and the dispatcher
    HB.shutdown(shutdown_workers=True)

    # pickle res
    with open(os.path.join(path, "hyperband_res.pkl"), 'w+b') as f:
        pickle.dump(res, f)
    # save incumbent trajectory as csv
    traj = res.get_incumbent_trajectory()
    incumbent_performance = traj["losses"]
    # incumbent_perf_dict = {"hyperband_incumbent_trajectory": incumbent_performance}
    
    best_cfg_id = res.get_incumbent_id()
    # print("best_cfg_id",best_cfg_id)
    all_configs = res.get_id2config_mapping()
    # print("all_configs",all_configs)
    
    best_cfg = all_configs[best_cfg_id]['config']
    print("return best_cfg: ",best_cfg)
    
    return best_cfg