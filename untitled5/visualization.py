# -*- coding: utf-8 -*-

import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import os
%matplotlib inline

FILE_PATH='D:/wuyu/untitled5/201802'

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 9,
}
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}
color=['red','yellow','green','blue','black','purple']
def stock_vis(da,ticker='600000.SH'):

    x = da['time'][:60].astype(str)
    print(x)
    plt.plot(x, da['ticker' == ticker, 0,:, 0][:60], color='red', label='Open', linewidth=1.1)
    plt.legend(loc='upper right', prop=font2, frameon=False)

    plt.plot(x, da['ticker' == ticker, 0,:, 1][:60], color='purple', label='High', linewidth=1.1)
    plt.legend(loc='upper right', prop=font2, frameon=False)

    plt.plot(x, da['ticker' == ticker, 0, :, 2][:60], color='blue', label='Low', linewidth=1.1)
    plt.legend(loc='upper right', prop=font2, frameon=False)

    plt.plot(x, da['ticker' == ticker, 0, :, 3][:60], color='green', label='High', linewidth=1.1)
    plt.legend(loc='upper right', prop=font2, frameon=False)
    #plt.xticks([])  # 去掉x坐标轴刻度
    plt.show()













if __name__== '__main__':
    for file in os.listdir(FILE_PATH):
        if file[-3:]=='.h5' and file[:4]== '2018':
            print(file[:-3])
    datetmp= raw_input("input the date:")
    dataarray= xr.open_dataarray(FILE_PATH+'/'+datetmp+'.h5')
    tickertmp= raw_input("input the ticker")
    stock_vis(dataarray,tickertmp)
