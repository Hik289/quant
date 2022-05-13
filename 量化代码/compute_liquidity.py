
# coding: utf-8

# In[ ]:


import math
import numpy as np
from numpy import *
import pandas as pd
import statsmodels.api as sm
from numpy import nan as NaN
import math
from pandas.core.frame import DataFrame
import os
import time


# In[ ]:


volume=pd.read_csv('./volume.csv').set_index('S_INFO_WINDCODE')
CCS=pd.read_csv('./circulation_capitalStock.csv').set_index('S_INFO_WINDCODE')
Return=pd.read_csv('./Return.csv').set_index('S_INFO_WINDCODE')

# In[ ]:


timelist=list(Return.columns)
stock_list=list(Return.index)
startindex=timelist.index('20000107')
endindex=timelist.index('20180831')
STOM = dict()
T = 21
for i in range(0, len(stock_list)):
    print('STOM, '+str(i))
    #stomtime = (pd.to_datetime(startdate) + timedelta(-3)).strftime('%Y%m%d')
    #startindexstom = timelist.index(stomtime)
    STOM[stock_list[i]] = dict()
    for t in range(startindex, endindex + 1, 1):
        data = volume.iloc[i, (t - T):t] / CCS.iloc[i, (t - T):t]
        #print(data)
        STOM[stock_list[i]][timelist[t]] = log(sum(data)) #只截取需要更新的时间片段
STOM1 = pd.DataFrame(STOM).T
where_are_inf = np.isinf(STOM1)
STOM1[where_are_inf] = NaN

T1 = 3
STOQ = dict()
for i in range(0, len(stock_list)):
    print('STOQ, '+str(i))
    STOQ[stock_list[i]] = dict()
    for t in range(startindex + T1, endindex + 1, 1): #+T1:往后算T1个
        data = exp(STOM1.iloc[i, (t - startindex - T1):(t - startindex)])
        STOQ[stock_list[i]][timelist[t]] = log(sum(data) / T1)
STOQ1 = pd.DataFrame(STOQ).T
where_are_inf = np.isinf(STOQ1)
STOQ1[where_are_inf] = NaN

T2 = 12
STOA = dict()
for i in range(0, len(stock_list)):
    print('STOA, '+str(i))
    STOA[stock_list[i]] = dict()
    for t in range(startindex + T2, endindex + 1, 1): #往后算12个
        data = exp(STOM1.iloc[i, (t - startindex - T2):(t - startindex)])
        STOA[stock_list[i]][timelist[t]] = log(sum(data) / T2)
STOA1 = pd.DataFrame(STOA).T
where_are_inf = np.isinf(STOA1)
STOA1[where_are_inf] = NaN

liquidity = 0.35 * STOM1 + 0.35 * STOQ1 + 0.30 * STOA1


# In[ ]:


liquidity.to_csv(('./liquidity_new.csv'))
