# -*- coding: utf-8 -*-
import os
import util
import numpy as np
import matplotlib.pyplot as plt

def genSignalPic(start, end, signalVer, dataDir, signalDir, outDir, prePeriod, fowPeriod, signalQt, bins=5):
    ################################
    #对于每种信号，根据指标排序的分位点确定long short阈值
    #Usage:
    #   -- start : 起始日期
    #   -- end : 结束日期
    #   -- signalVer : 信号版本
    #   -- dataDir : 数据目录
    #   -- signalDir : 信号目录
    #   -- outDir : 输出目录
    #   -- prePeriod : 回溯时间长度
    #   -- fowPeriod : 展期时间长度
    #   -- signalQt : 分位点
    #   -- bins : Tprofile 中 timeList 的个数
    ################################
    if os.path.exists(os.path.join(outDir, signalVer)) == False:
        os.makedirs(os.path.join(outDir, signalVer))
    dateRange = util.getTradingDayRange(start, end)
    mktData = util.readDateRangeData(dateRange, dataDir)
    rollMktData = mktData.rolling(time=prePeriod+fowPeriod).construct('period').shift(time=-fowPeriod)
    rollMktData['period'] = np.arange(-prePeriod+1, fowPeriod+1)
    sigData = util.readDateRangeData(dateRange, os.path.join(signalDir, signalVer))
    timeList = sigData['time'][(prePeriod-1):-fowPeriod]
    if len(timeList) > bins:
        timeList = timeList[::len(timeList)//bins][-bins:]
    termList = sigData['variable']
    for term in termList.values:
        md = rollMktData.sel(time=timeList.values, variable='Return')
        sd = sigData.sel(time=timeList.values, variable=term)
        longQt = sd.quantile(signalQt, dim='ticker')
        shortQt = sd.quantile(1-signalQt, dim='ticker')
        longTProfile = md.where(sd > longQt).mean(dim=['ticker', 'date'])
        shortTProfile = md.where(sd < shortQt).mean(dim=['ticker', 'date'])
        (1+longTProfile).cumprod(dim='period').plot.line(x='period', hue='time', figsize=(12, 7))
        plt.savefig(os.path.join(outDir, signalVer, term+'_longSignal.png'))
        (1+shortTProfile).cumprod(dim='period').plot.line(x='period', hue='time', figsize=(12, 7))
        plt.savefig(os.path.join(outDir, signalVer, term+'_shortSignal.png'))
            
    
if __name__ == '__main__':
    startDate = 20180101
    endDate = 20180111
    dataDir = '../data/md/minute'
    signalDir = '../data/signal'
    signalVer = 'DEMO'
    outDir = '../report'
    field = 'Return'
    prePeriod = 15
    fowPeriod = 15
    signalQt = .99
    genSignalPic(start=startDate, end=endDate, signalVer=signalVer, 
                 dataDir=dataDir, signalDir=signalDir, outDir=outDir, 
                 prePeriod=prePeriod, fowPeriod=fowPeriod, signalQt=signalQt)