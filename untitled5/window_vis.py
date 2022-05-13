# -*- coding: utf-8 -*-

import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, Formatter
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as dates
import matplotlib.ticker as mticker
import os
import pandas as pd
import numpy as np

SIGNAL_FILE_PATH='D:/wuyu/untitled5/signal'
STOCK_FILE_PATH='D:/wuyu/untitled5/201802'


font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 9,
}
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}
colors=['red','yellow','green','blue','black','purple']
'''
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
    
'''


class StFormatter(Formatter):
    def __init__(self, dates, fmt='%Y%m%d %H:%M'):
        self.dates = dates
        self.fmt = fmt

    def __call__(self, x, pos=0):

        ind = int(np.round(x))

        return dates.num2date(ind / 1440).strftime(self.fmt)


def get_time_ready(df):
    df2=[]
    for date in df:
        tmp=date[0]/100
        k= str(int(tmp%100))
        if len(k) ==1:
            k='0'+k
        tmp='2018/01/02-'+str(int((tmp- tmp%100)/100))+':'+ k
        print(tmp)
        df2.append([tmp,date[1],date[2],date[3],date[4],date[5]])
    df2 = pd.DataFrame(data=df2, columns=['time', 'Open', 'High', 'Low', 'Close', 'Volume'])

    df2['time']= pd.to_datetime(df2['time'],format= "%Y/%m/%d-%H:%M")

    df2['time'] = df2['time'].apply(lambda x: dates.date2num(x) * 1440)

    return df2


def signal_vis(signal_data, stock_data):
    stock_data[stock_data['Volume']== 0]= np.nan
    stock_data=stock_data.dropna() #去掉成交量为0的点

    '''
        x=0
        y=len(signal_data.time)
        ohlc=[]

        while x< y:
            append_me = signal_data['time'][x], signal_data['Open'][x], signal_data['High'][x], signal_data['Low'][x], signal_data['Close'][x], signal_data['Volume'][x]
            ohlc.append(append_me)
            x+= 1
        fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(1200 / 72, 480 / 72))
        fig.subplots_adjust(bottom=0.1)
        candlestick_ochl(ax1, ohlc, colordown='#53c156', colorup='#ff1717')
        ax1.grid(True)
        plt.bar(ohlc[:, 0], ohlc[:, 5], width=200)
        ax2.set_ylabel('Volume')
        ax2.grid(True)

        plt.show()
    '''
    width= 0.2
    dm= stock_data[['time','Open','High','Low','Close','Volume']].values
    dm=get_time_ready(dm).values

    fig, (ax1,ax2) = plt.subplots(2, sharex=True,figsize=(1200/72,480/72))
    fig.subplots_adjust(bottom= 0.1)



    candlestick_ohlc(ax1, dm, colordown='#53c156', colorup='#ff1717', width=width, alpha=1)
    ax1.grid(True)
    ax1.xaxis_date()
    plt.bar(dm[:, 0], dm[:, 5], width=200)
    ax2.set_ylabel('Volume')
    ax2.grid(True)
    '''for s in sg:
        print(s)
        if s[2]== 1.0:
            plt.annotate(u"这是买入信号", xy=(s.values[0][1], s.loc[s.index[s.time==signal_data.values[0][1]]].values[4]), xytext=(-4, 50), arrowprops=dict(facecolor="r", headlength=10, headwidth=30, width=20))
        if s[2]== -1.0:
            plt.annotate(u"这是卖出信号", xy=(s.values[0][1], s.loc[s.index[s.time == signal_data.values[0][1]]].values[4]),
                         xytext=(4, 50), arrowprops=dict(facecolor="g", headlength=10, headwidth=30, width=20))
'''

    formatter = StFormatter(dm[:, 0])
    ax2.xaxis.set_major_formatter(formatter)
    for label in ax2.get_xticklabels():
        label.set_rotation(90)
        label.set_horizontalalignment('right')

    plt.show()

'''
if __name__== '__main__':
    for file in os.listdir(FILE_PATH):
        if file[-3:]=='.h5' and file[:4]== '2018':
            print(file[:-3])
    datetmp= raw_input("input the date:")
    dataarray= xr.open_dataarray(FILE_PATH+'/'+datetmp+'.h5')
    tickertmp= raw_input("input the ticker")
    stock_vis(dataarray,tickertmp)
'''




def readstkdata(rootpath= STOCK_FILE_PATH, signalpath= SIGNAL_FILE_PATH, stime= 93000, etime= 150000):

    f= os.listdir(signalpath)[0]
    stock_file = rootpath + '/' + f[:-3] + 'h5'

    df= pd.read_csv(signalpath+'/'+f)
    print(df['ticker'])
    #ticker= raw_input('input ticker')
    ticker='000001.SZ'

    df2= df[df.ticker== ticker]

    sd= xr.open_dataarray(stock_file)
    sd=sd.sel(ticker= ticker)[0].loc[stime:etime,:]
    sd= sd.to_pandas().reset_index()
    return df2, sd



if __name__== '__main__':
    m= readstkdata()

    signal_vis(m[0], m[1])