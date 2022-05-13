# -*- coding: utf-8 -*-
from handledata import *
import xarray as xr
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as pp


file_dir= "C:\Users\ThinkPad\Desktop"
file_name= '20180102.h5'

def genDEMOSignalHelper(trigger, date, files, dateWindow=1, timeWindow=15, min_period=15):
    if dateWindow == 1:
        data = files
    '''else:
        data = hqUtil.readDailyData(date, dataDir)'''
    ## data 是包含 ticker, date, time, variable 四个维度的 xarray.DataArray
    roll = data.rolling(time=timeWindow, min_periods=min_period).construct('timeNum')

    ## 调用 construct 方法之后由于每个时间点上有 timeWindow 的数据所以多出一个 timeNum 的维度
    def handleBar(Bar):
        vol = Bar.sel(variable='Volume')
        ret = Bar.sel(variable='Return')
        volWgt = vol / vol.sum(dim='timeNum')
        weightedReturnMean = (ret * volWgt).sum(dim='timeNum')
        stdAdjCoef = 1 / (1 - 1 / ret.count(dim='timeNum'))
        weightedReturnStd = np.sqrt(stdAdjCoef * (
                (volWgt * ret * ret).sum(dim='timeNum') - (volWgt * ret).sum(dim='timeNum') ** 2))
        term = (weightedReturnMean / weightedReturnStd).expand_dims(['variable', 'time']).transpose('ticker', 'date',
                                                                                                    'time', 'variable')
        term.coords['variable'] = ['termDEMO']

        term = term.fillna(0) * (~np.isinf(term)).astype(int)
        buyThresh = trigger
        selThresh = -0.8* trigger


        return (term > buyThresh).astype(int) - (term < selThresh).astype(int)

    def handleBar2(Bar):
        high = Bar.sel(variable="High")
        low = Bar.sel(variable="Low")
        highMean = high.sum(dim='timeNum')/min_period
        LowMean = low.sum(dim='timeNum')/min_period
        stdHighAdjCoef = 1 / (1 - 1 / high.count(dim='timeNum'))
        stdLowAdjCoef = 1 / (1 - 1 / low.count(dim='timeNum'))
        highStd = np.sqrt(stdHighAdjCoef * (
                (high *high).sum(dim='timeNum') - high.sum(dim='timeNum') ** 2))
        lowStd = np.sqrt(stdLowAdjCoef * (
                (low *low).sum(dim='timeNum') - low.sum(dim='timeNum') ** 2))
        factor= 1.0
        highterm = (highMean /(factor *highStd)).expand_dims(['variable']).transpose('ticker', 'date','time','variable')
        lowterm = (lowMean /(factor *lowStd)).expand_dims(['variable']).transpose('ticker', 'date', 'time', 'variable')

        highterm.coords['variable'] = ['termDEMO']
        lowterm.coords['variable'] = ["termDEMO"]

        highterm = highterm.fillna(0) * (~np.isinf(highterm)).astype(int)
        lowterm = lowterm.fillna(0) * (~np.isinf(lowterm)).astype(int)


    sigs = xr.concat([handleBar(roll[:, :, t, :, :]) for t in range(min_period - 1, len(roll['time']))], dim='time')
    pp.plot(data["time"][14:],sigs[1,0,:,0])
    pp.show()
    return sigs

def dealDignal(sigs):
    sigs_modify= sigs
    return sigs_modify


if __name__=='__main__':
    files= xr.open_dataarray(file_dir+"\\"+file_name)
    print(files)
    datewindow= 1
    trigger= 0.7
    timewindow= 15
    mini_period= 15
    sigs= genDEMOSignalHelper(trigger, files[:-4], files, dateWindow= datewindow, timeWindow= timewindow, min_period= mini_period)
    pp.plot(files["time"][14:],sigs[1,0,:,0])
    pp.show()
    pp.scatter(files["time"], files.sel(variable='Return')[0,0,:])
    pp.show()
    print(sigs)

