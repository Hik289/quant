# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 09:22:11 2018

@author: wufei
"""

import pandas as pd
import numpy as np
from pandas import DataFrame
import time
import statsmodels.api as sm
import os
import matplotlib as mpl
mpl.use('Agg')
#import matplotlib.pyplot as plt
from datetime import datetime,timedelta

def read_alpha_list(alpha_list):
    alphas=[]
    for x in alpha_list:
        alphas.append(pd.read_csv('/usr/intern/wufei/fun_factor/alpha%s.csv'%x,index_col='S_INFO_WINDCODE'))
    return alphas

def regression(x,y):
    ols_model=sm.OLS(y,sm.add_constant(x))
    return ols_model.fit()


def getdataready(timelist_column,alpha):
    data=pd.concat([RET_new[timelist_column],alpha[timelist_column]],axis=1)
    data.iloc[:,1]=data.iloc[:,1].fillna(data.iloc[:,1].mean())
    data=data.dropna()
    alpha_new=data.iloc[:,1]
    alpha_new=(alpha_new-alpha_new.mean())/alpha_new.std()
    data.iloc[:,1]=alpha_new
    return data
def drawpicture(data):
    plotdata=data
    plotdata=(1+plotdata).cumprod(0)
    plotdata.index=pd.to_datetime(data_f.columns)
    plotdata.plot()
    mpl.pyplot.savefig('/usr/intern/wufei/fun_factor/hdj/'+plotdata.columns[0]+'.png')
    mpl.pyplot.close('all')
def dropextreme(start_point,end_point,timelist,alpha):
    for j in range(start_point,end_point+1):
        date=timelist[j]
        a=np.abs(alpha[date]-alpha[date].median())
        diff=a.median()
        maxrange=alpha[date].median()+4*diff
        minrange=alpha[date].median()-4*diff
        alpha[date]=alpha[date].clip(minrange,maxrange)
    return alpha
    
start_date_time = '20110101'
end_date_time = '20180801'
alpha_list=list(range(1,11))
alphas=read_alpha_list(alpha_list)
RET=pd.read_csv('/usr/intern/wufei/fun_factor/risk factors/Return.csv',index_col='S_INFO_WINDCODE')
timelist=list(RET.columns)
RET_new=RET.shift(periods=-20,axis=1)
RET_new=RET_new.rolling(window=20,axis=1).sum()
while start_date_time not in timelist:
    start_date_time=(pd.to_datetime(start_date_time)+timedelta(1)).strftime('%Y%m%d')
while end_date_time not in timelist:
    end_date_time=(pd.to_datetime(end_date_time)+timedelta(-1)).strftime('%Y%m%d')
start_point=timelist.index(start_date_time)
end_point=timelist.index(end_date_time)-3

data_f=pd.DataFrame()
data_t=pd.DataFrame()
data_ic=pd.DataFrame()

for i in range(0,len(alpha_list)):
    alpha=dropextreme(start_point,end_point,timelist,alphas[i])
    for j in range(start_point,end_point+1):
        if j%20==0:
            data=getdataready(timelist[j],alpha)
            params=regression(data.iloc[:,1],data.iloc[:,0]).params
            tvalues=regression(data.iloc[:,1],data.iloc[:,0]).tvalues
            data_f.loc['alpha'+str(alpha_list[i]),timelist[j]]=params[1]
            data_t.loc['alpha'+str(alpha_list[i]),timelist[j]]=tvalues[1]
            data_ic.loc['alpha'+str(alpha_list[i]),timelist[j]]=np.corrcoef(data.iloc[:,0],data.iloc[:,1])[0][1]
data_f.to_csv('data_f.csv')
data_t.to_csv('data_t.csv')
data_ic.to_csv('data_ic.csv')
for i in range(0,len(alpha_list)):
    draw=data_f.iloc[i,:]
    draw=pd.DataFrame(data=draw.T.values,columns=['alpha'+str(i)],index=data_f.columns)
    drawpicture(draw)
    
    
    

