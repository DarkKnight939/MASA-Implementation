#！/usr/bin/python
# -*- coding: utf-8 -*-#

'''
---------------------------------
 Name: entrance.py  
 Author: MASA
--------------------------------
'''

import os
os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import random 
import numpy as np
import torch as th
import datetime
import copy
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
th.use_deterministic_algorithms(True)

import pandas as pd
import time
from config import Config
from utils.featGen import FeatureProcesser
from utils.tradeEnv import StockPortfolioEnv, StockPortfolioEnv_cash
from utils.model_pool import model_select, benchmark_algo_select
from utils.callback_func import PoCallback
from RL_controller.market_obs import MarketObserver, MarketObserver_Algorithmic
from utils.data_downloader import download_market_data
import timeit

def RLonly(config):
    # For running the single-agent RL-based framework (TD3-Profit, TD3-PR, TD3-SR)
    # Get dataset
    # Allow using an explicit dataset file if provided in the config (or via MASA_DATA_FILE env var)
    if getattr(config, 'dataFile', None):
        fpath = config.dataFile
    else:
        mkt_name = config.market_name
        fpath = os.path.join(config.dataDir, '{}_{}_{}.csv'.format(mkt_name, config.topK, config.freq))

    if not os.path.exists(fpath):
        # Attempt to download the required dataset automatically (only for auto-constructed filenames)
        # If the user specified an explicit dataFile, don't attempt auto-download — fail so the user can correct the path.
        if getattr(config, 'dataFile', None):
            raise ValueError("Cannot load the data from {}. Please check the path provided in Config.dataFile".format(fpath))
        print(f"Dataset {fpath} not found. Attempting to download using yfinance...")
        try:
            download_market_data(config)
        except Exception as e:
            raise ValueError("Cannot load the data from {} and automatic download failed: {}".format(fpath, e))
        if not os.path.exists(fpath):
            raise ValueError("Cannot load the data from {} after downloading".format(fpath))

    data = pd.DataFrame(pd.read_csv(fpath, header=0))
    
    # Ensure market index file has required columns; if not, try to (re)download
    idx_path = os.path.join(config.dataDir, '{}_{}_index.csv'.format(mkt_name, config.freq))
    required_idx_cols = set(['date', 'open', 'high', 'low', 'close'])
    try:
        if os.path.exists(idx_path):
            with open(idx_path, 'r') as f:
                header = f.readline().strip().split(',')
            if not required_idx_cols.issubset(set(header)):
                print(f"Existing index file {idx_path} doesn't contain required market columns; regenerating via downloader.")
                download_market_data(config)
        else:
            # index file missing -> attempt download
            download_market_data(config)
    except Exception as e:
        print(f"Warning: automatic index regeneration failed: {e}")

    # Preprocess features
    featProc = FeatureProcesser(config=config)
    data_dict = featProc.preprocess_feat(data=data)
    tech_indicator_lst = featProc.techIndicatorLst 
    stock_num = data_dict['train']['stock'].nunique()
    print("Data has been processed..")

    # Initialize environment
    
    trainInvest_env_para = config.invest_env_para 
    env_train = StockPortfolioEnv(
        config=config, rawdata=data_dict['train'], mode='train', stock_num=stock_num, action_dim=stock_num, 
        tech_indicator_lst=tech_indicator_lst, **trainInvest_env_para
    )
    if (config.valid_date_start is not None) and (config.valid_date_end is not None):
        validInvest_env_para = config.invest_env_para 
        env_valid = StockPortfolioEnv(
            config=config, rawdata=data_dict['valid'], mode='valid', stock_num=stock_num, action_dim=stock_num, 
            tech_indicator_lst=tech_indicator_lst, **validInvest_env_para
        )
    else:
        env_valid = None

    if (config.test_date_start is not None) and (config.test_date_end is not None):
        testInvest_env_para = config.invest_env_para 
        env_test = StockPortfolioEnv(
            config=config, rawdata=data_dict['test'], mode='test', stock_num=stock_num, action_dim=stock_num, 
            tech_indicator_lst=tech_indicator_lst, **testInvest_env_para
        )
    else:
        env_test = None

    # Load RL model
    ModelCls = model_select(model_name=config.rl_model_name, mode=config.mode)
    model_para_dict = config.model_para
    po_model = ModelCls(env=env_train, **model_para_dict) # Create instance 
    total_timesteps = int(config.num_epochs * env_train.totalTradeDay)
    print('Training Start', flush=True)
    log_interval = 10
    callback1 =PoCallback(config=config, train_env=env_train, valid_env=env_valid, test_env=env_test)
    cpt_start = time.process_time()
    perft_start = time.perf_counter()
    timeit_default = timeit.default_timer()
    my_globals = globals()
    my_globals.update({'po_model': po_model, 'total_timesteps': total_timesteps, 'callback1': callback1, 'log_interval': log_interval})
    t = timeit.Timer(stmt='po_model.learn(total_timesteps=total_timesteps, callback=callback1, log_interval=log_interval)', globals=my_globals)
    time_usage = t.timeit(number=1)
    cpt_usgae = time.process_time() - cpt_start
    perf_usgae = time.perf_counter() - perft_start
    timeit_usgae = timeit.default_timer() - timeit_default
    print("Time usgae for {} epochs: {}s, cpu time: {}s, perf_couter: {}s, timeit_default: {}s".format(config.num_epochs, np.round(time_usage, 2), np.round(cpt_usgae, 2), np.round(perf_usgae, 2), np.round(timeit_usgae, 2)))
    print("-*"*20)

    del po_model
    print("Training Done...", flush=True)


def RLcontroller(config):
    # For running the MASA framework
    # Get dataset
    # Allow using an explicit dataset file if provided in the config (or via MASA_DATA_FILE env var)
    if getattr(config, 'dataFile', None):
        fpath = config.dataFile
    else:
        mkt_name = config.market_name
        fpath = os.path.join(config.dataDir, '{}_{}_{}.csv'.format(mkt_name, config.topK, config.freq))

    if not os.path.exists(fpath):
        if getattr(config, 'dataFile', None):
            raise ValueError("Cannot load the data from {}. Please check the path provided in Config.dataFile".format(fpath))
        print(f"Dataset {fpath} not found. Attempting to download using yfinance...")
        try:
            download_market_data(config)
        except Exception as e:
            raise ValueError("Cannot load the data from {} and automatic download failed: {}".format(fpath, e))
        if not os.path.exists(fpath):
            raise ValueError("Cannot load the data from {} after downloading".format(fpath))

    data = pd.DataFrame(pd.read_csv(fpath, header=0))

    # Ensure market index file has required columns; if not, try to (re)download
    idx_path = os.path.join(config.dataDir, '{}_{}_index.csv'.format(mkt_name, config.freq))
    required_idx_cols = set(['date', 'open', 'high', 'low', 'close'])
    try:
        if os.path.exists(idx_path):
            with open(idx_path, 'r') as f:
                header = f.readline().strip().split(',')
            if not required_idx_cols.issubset(set(header)):
                print(f"Existing index file {idx_path} doesn't contain required market columns; regenerating via downloader.")
                download_market_data(config)
        else:
            # index file missing -> attempt download
            download_market_data(config)
    except Exception as e:
        print(f"Warning: automatic index regeneration failed: {e}")

    # Preprocess features
    featProc = FeatureProcesser(config=config)
    data_dict = featProc.preprocess_feat(data=data)
    tech_indicator_lst = featProc.techIndicatorLst
    stock_num = data_dict['train']['stock'].nunique()
    print("Data has been processed..")

    if config.enable_market_observer:
        if ('ma' in config.mktobs_algo) or ('dc' in config.mktobs_algo):
            mkt_observer = MarketObserver_Algorithmic(config=config, action_dim=stock_num)
        else:
            mkt_observer = MarketObserver(config=config, action_dim=stock_num) 
    else:
        mkt_observer = None

    # Initialize environment
    if (config.valid_date_start is not None) and (config.valid_date_end is not None):
        validInvest_env_para = config.invest_env_para 
        env_valid = StockPortfolioEnv(
            config=config, rawdata=data_dict['valid'], mode='valid', stock_num=stock_num, action_dim=stock_num, 
            tech_indicator_lst=tech_indicator_lst, extra_data=data_dict['extra_valid'], 
            mkt_observer=mkt_observer, **validInvest_env_para
        )
    else:
        env_valid = None
        raise ValueError("No validation set is provided for training")
    if (config.test_date_start is not None) and (config.test_date_end is not None):
        testInvest_env_para = config.invest_env_para 
        env_test = StockPortfolioEnv(
            config=config, rawdata=data_dict['test'], mode='test', stock_num=stock_num, action_dim=stock_num, 
            tech_indicator_lst=tech_indicator_lst, extra_data=data_dict['extra_test'], 
            mkt_observer=mkt_observer, **testInvest_env_para
        )
    else:
        env_test = None
        raise ValueError("No test set is provided for training")

    ModelCls = model_select(model_name=config.rl_model_name, mode=config.mode)
    # Initialize environment
    trainInvest_env_para = config.invest_env_para 
    env_train = StockPortfolioEnv(
        config=config, rawdata=data_dict['train'], mode='train', stock_num=stock_num, action_dim=stock_num, 
        tech_indicator_lst=tech_indicator_lst, extra_data=data_dict['extra_train'], 
        mkt_observer=mkt_observer, **trainInvest_env_para
    )

    # Load RL model
    model_para_dict = config.model_para
    po_model = ModelCls(env=env_train, **model_para_dict) 
    total_timesteps = int(config.num_epochs * env_train.totalTradeDay)
    print('Training Start', flush=True)
    log_interval = 10
    callback1 = PoCallback(config=config, train_env=env_train, valid_env=env_valid, test_env=env_test)
    cpt_start = time.process_time()
    perft_start = time.perf_counter()
    timeit_default = timeit.default_timer()
    my_globals = globals()
    my_globals.update({'po_model': po_model, 'total_timesteps': total_timesteps, 'callback1': callback1, 'log_interval': log_interval})
    t = timeit.Timer(stmt='po_model.learn(total_timesteps=total_timesteps, callback=callback1, log_interval=log_interval)', globals=my_globals)
    time_usage = t.timeit(number=1)
    cpt_usgae = time.process_time() - cpt_start
    perf_usgae = time.perf_counter() - perft_start
    timeit_usgae = timeit.default_timer() - timeit_default

    print("Time usgae for {} epochs: {}s, cpu time: {}s, perf_couter: {}s, timeit_default: {}s".format(config.num_epochs, np.round(time_usage, 2), np.round(cpt_usgae, 2), np.round(perf_usgae, 2), np.round(timeit_usgae, 2)))
    print("-*"*20)
    del po_model
    print("Training Done...", flush=True)

def entrance():
    """
    Entrance function for running the MASA framework
    """
    current_date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    rand_seed = int(time.mktime(datetime.datetime.strptime(current_date, '%Y-%m-%d-%H-%M-%S').timetuple()))

    random.seed(rand_seed)
    os.environ['PYTHONHASHSEED'] = str(rand_seed)
    np.random.seed(rand_seed)
    th.manual_seed(rand_seed)
    th.cuda.manual_seed(rand_seed)
    th.cuda.manual_seed_all(rand_seed)

    start_cputime = time.process_time()
    start_systime = time.perf_counter()
    config = Config(seed_num=rand_seed, current_date=current_date) 
    config.print_config()
    if config.mode == 'RLonly':
        RLonly(config=config)
    elif config.mode == 'RLcontroller':
        RLcontroller(config=config)
    elif config.mode == 'Benchmark':
        raise NotImplementedError("Please refer to the corresponding papers and their provided implementations.")
    else:
        raise ValueError('Unexpected mode {}'.format(config.mode))

    end_cputime = time.process_time()
    end_systime = time.perf_counter()
    print("[Done] Total cputime: {} s, system time: {} s".format(np.round(end_cputime - start_cputime, 2), np.round(end_systime - start_systime, 2)))
    
def main():
    entrance()

if __name__ == '__main__':
    main()
