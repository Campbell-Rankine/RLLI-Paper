import pandas as pd
import numpy as np
from functools import partial
import logging

import ray

import yahoo_fin.stock_info as si
from tqdm import tqdm

from Data.data_utils import *
import pickle
import os
from pathlib import Path
from collections import defaultdict

RAY_IGNORE_UNHANDLED_ERRORS = 1
import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})
### - Distributed Remote Processing - ###
@ray.remote(max_retries=5)
def download_ticker(ticker,read_range):
    ### - Download ticker data - ###
    print(ticker)
    try:
        tick = si.get_data(ticker, read_range[1], read_range[0])
        ### - Process Data, Extract indicators - ###
        tick.index.names = ['date']
        tick.drop('ticker', axis=1)
        tick.columns = [col.lower() for col in tick.columns]
        #tick = generate_features(tick).dropna(axis=1)
        return tick
    except Exception:
        pass

@ray.remote(max_retries=5)
def download_ticker_ind(ticker,read_range):
    ### - Download ticker data - ###
    print(ticker)
    try:
        tick = si.get_data(ticker, read_range[1], read_range[0])
        ### - Process Data, Extract indicators - ###
        tick.index.names = ['date']
        tick.drop('ticker', axis=1)
        tick.columns = [col.lower() for col in tick.columns]
        tick = generate_features(tick).dropna(axis=1)
        return tick
    except Exception:
        pass

def build_dataset_serial(debug):
    tickers = si.tickers_sp500()

    read_range = ['2022-11-1','2019-7-12']

    print('build dataset')
    write = defaultdict([])
    databar = tqdm(enumerate(tickers))
    if debug:
        databar = tqdm(enumerate(tickers[0:3]))
    for i, ticker in databar:
        tick = si.get_data(ticker, read_range[1], read_range[0])
        ### - Process Data, Extract indicators - ###
        tick.index.names = ['date']
        tick.columns = [col.lower() for col in tick.columns]
        tick = generate_features(tick).dropna(axis=1)
        write[ticker] = tick
    if not write == None:
        return write

def build_dataset(tickers, debug):
    ### - Save Info TODO: Add to args - ###
    ### - TODO: remove Ray and try and code properly from scratch, then reinsert Ray code - ###
    read_range = ['2022-11-1','2019-7-12']

    ### - Init Ray - ###
    print("init ray")
    ray.init()
    print('build dataset')

    if debug:
        tickers = tickers[0:3]

    data = [download_ticker.remote(tick, read_range) for tick in tickers]
    try:
        write = ray.get(data)
    except Exception:
        print('... Error: most likely invalid Ticker ...')

    if not write == None:
        return write

def build_dataset_ind(tickers, debug):
    ### - Save Info TODO: Add to args - ###
    ### - TODO: remove Ray and try and code properly from scratch, then reinsert Ray code - ###
    read_range = ['2022-11-1','2019-7-12']

    ### - Init Ray - ###
    print("init ray")
    ray.init()
    print('build dataset')

    if debug:
        tickers = tickers[0:3]

    data = [download_ticker_ind.remote(tick, read_range) for tick in tickers]
    try:
        write = ray.get(data)
    except Exception:
        print('... Error: most likely invalid Ticker ...')

    if not write == None:
        return write

def download_build(to_file):
    ### - Get Ticker List - ###
    tickers = si.tickers_sp500()
    
    args = process_command_line_arguments()
    ### - Debug Flag, Add to Args - ###
    debug = args.debug

    data = build_dataset(tickers, debug)
    data = dict(zip(tickers, data))
    print(data)
    ### - File Saving - ###
    if not os.path.exists(Path(to_file)):
        Path(to_file).mkdir(exist_ok=False)
    
    with open(to_file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def download_build_ind(to_file):
    ### - Get Ticker List - ###
    tickers = si.tickers_sp500()
    
    args = process_command_line_arguments()
    ### - Debug Flag, Add to Args - ###
    debug = args.debug

    data = build_dataset_ind(tickers, debug)
    data = dict(zip(tickers, data))
    print(data)
    ### - File Saving - ###
    if not os.path.exists(Path(to_file)):
        Path(to_file).mkdir(exist_ok=False)
    
    with open(to_file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    args = process_command_line_arguments()
    if not args.ind:
        to_file = 'dataset_long_ohlc.pickle'
        download_build(to_file = to_file)
    else:
        to_file = 'dataset_long_ind.pickle'
        download_build_ind(to_file = to_file)