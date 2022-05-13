# -*- coding: utf-8 -*-
import os
import socket
import numpy as np
import pandas as pd
import xarray as xr
import tushare as ts


def getTradingDayRange(st, ed, timeWindow=1, rolling=False, delay=0):
    date = ts.trade_cal()
    tradeDate = date[date.isOpen==1].calendarDate.apply(lambda x:x.replace('-', '')).astype(int)
    endIndex = tradeDate[np.array(tradeDate <= ed)].index[-1]
    if st == ed:
        startIndex = endIndex
    else:
        startIndex = tradeDate[np.array(tradeDate >= st)].index[0]
    if (timeWindow - 1) :
        startIndex -= (timeWindow - 1)
    if delay :
        startIndex -= delay
        endIndex -= delay
    dateRange = tradeDate.loc[startIndex:endIndex]
    return dateRange.values.flatten()

def handle_minuteBar(data, market):
    columns = ['ticker', 'DateTime', 'PreClose', 'Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'LongPosition']
    data.columns = columns
    data['ticker'] = data.ticker.apply(lambda x : str(x).zfill(6)+'.'+market)
    data['date'] = data.DateTime // 1000000
    data['time'] = data.DateTime % 1000000
    if market == 'SH':
        dataAShare = data[data.ticker.apply(lambda x:x.startswith('6'))].copy()
    elif market == 'SZ':
        dataAShare = data[~data.ticker.apply(lambda x:x.startswith('399') | x.startswith('200'))].copy()
    dataAShare['Return'] = 0
    dataXr = dataAShare.set_index(['ticker', 'date', 'time']).loc[:, ['Open', 'High', 'Low', 'Close', 'PreClose', 'Return', 'Volume', 'Amount']].to_xarray().to_array().transpose('ticker', 'date', 'time', 'variable')
    dataXr.loc[:, :, 93100:, 'PreClose'] = dataXr.loc[:, :, :145900, 'Close'].values
    dataXr.loc[:, :, :, 'Return'] = ret = (dataXr.loc[:, :, :, 'Close']/dataXr.loc[:, :, :, 'PreClose']) - 1
    return dataXr

