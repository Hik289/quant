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

'''
host = '114.80.63.137'
port = 2222
usr = 'mds'
key = 'Mds12345!@#$%'


## 行情回放请求
FASTmap = {'BeginString':8, 
           'BodyLength':9, 
           'CheckSum':10, 
           
           'MsgType':35, 
           'SecderCompID':49,  
           'TargetCompID':56, 
           'MsgSeqNum':34, 
           'SendingTime':52, 
           'MessageEncoding':347, 
           
           ## Logon MsgType = A
           'EncryptMethod':98, # 0
           'HeartBtInt':108, 
           'AppID':553, 
           'UsernID':554, 
           
           ## Logout MsgType = 5
           58:'Text', 
           
           ## 行情回放 MsyType = UA1201
           10075:'RebuildMethod', # 4
           207:'SecurityExchange', # 1:SH
           1500:'MDStreamID', # 1 for level1 and 2 for level2
           10003:'OrigDate', 
           42:'OrigTime', 
           10076:'PlaybackSpeed'}

host = '172.23.1.177'
port = 6808
### Msg 的 Value 需要是 string
logonMsg = {'BeginString':'STEP.1.0.0', 
            'BodyLength':0,
            'MsgType':'A', 
            'UsernID':0, 
            'CheckSum':0}
logonMsg['BodyLength'] = str(len(logonMsg.keys()) - 3)
logonMsg['CheckSum'] = str(len(''.join(logonMsg.values())) % 252).zfill(3)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((host, port))
s.send(r'<SOH>'.join([str(FASTmap[k])+'='+v for k, v in logonMsg.items()]))
'''

if __name__ == '__main__':
    marketVer = ['sh2', 'sz2']
    dataDir = '/mnt/hisdata'
    dataVer = 'binary'
    outDir = '~/data/md/minute2'
    if os.path.exists(outDir)==False:
        os.makedirs(outDir)
    dateRange = getTradingDayRange(20180101,20180531)
    for d in dateRange.astype(str):
        print('reading '+d+' data ...')
        data = xr.concat([handle_minuteBar(pd.read_csv(os.path.join(dataDir, dataVer, m, d, 'Minute.csv'), skiprows=1, header=None), m[:2].upper()) for m in marketVer], dim='ticker')
        data.to_netcdf(os.path.join(outDir, d+'.h5'))