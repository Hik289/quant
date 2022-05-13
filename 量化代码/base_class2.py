import numpy as np
from numpy import *
import pandas as pd
import os
#import matplotlib as mpl
#mpl.use('Agg') # 如果在服务器上跑，且需要画图，这两行不能少，否则会报错
import statsmodels.api as sm
from config import *
import config


class base_class():

    def __init__(self, risk_factor_list, alpha_list=None):

        self.risk_factor_list = risk_factor_list
        self.alpha_list = alpha_list
        start = time.time()
        self.risk_factors = self.read_risk_factors()
        print('Time used for reading risk factors: ', time.time()-start)
        start = time.time()
        self.read_other_factors()
        print('Time used for reading other factors: ', time.time()-start)
        
    def read_other_factors(self):  #默认顺序是：Return、industry、weight

        self.other_factors = []
        other_list = ['Return','industry','weight']
        
        for i in range(0,len(other_list)):
            names = locals()
            names[other_list[i]] = pd.read_csv(other_factor_path + '%s.csv'%other_list[i],sep=None,engine='python')
            #names[other_list[i]] = pd.read_csv(other_factor_path + '/%s.csv'%other_list[i],sep=None)
            self.other_factors.append(names[other_list[i]])

        for x in self.other_factors:
            x = x.set_index('S_INFO_WINDCODE',inplace=True)
        self.timelist = list(self.other_factors[0].columns)
        self.index = list(self.other_factors[0].index)
        
    def read_risk_factors(self):

        risk_factors = []
        
        for i in range(0,len(self.risk_factor_list)):
            names = locals()
            names[self.risk_factor_list[i]] = pd.read_csv(risk_path + '%s.csv'%self.risk_factor_list[i],sep=None,engine='python')
            #names[self.risk_factor_list[i]] = pd.read_csv(risk_path + '/%s.csv'%self.risk_factor_list[i],sep=None)
            risk_factors.append(names[self.risk_factor_list[i]])

        for x in risk_factors:
            x = x.set_index('S_INFO_WINDCODE',inplace=True)

        return risk_factors


    def read_alphas(i): #这里的i和alpha文件名一一对应，表示读取文件名为alphai的文件。此处的read alphas函数专门写成了静态函数而不是实例的函数，这样在pick alpha里不需要实例化也可以调用这个函数
          
        alpha = pd.read_csv(config.alpha_path + '/alpha%d.csv'%i,sep=None,engine='python')
        alpha = alpha.set_index('S_INFO_WINDCODE')

        return alpha

    def getDataReady(self, timelist_columns, d, only_factor, *alphas): #d为预测周期，timelist_columns的格式为‘20090803’

        other_columns = [] 

        for x in alphas:
            if not x.empty:
                other_columns.append(x[timelist_columns])
            else:
                print('factor %s on %d is empty' % (alphas.index(x),timelist_columns))
            
        for x in self.risk_factors:
            other_columns.append(x[timelist_columns])

        other_columns_df = pd.concat(other_columns, axis=1)
        
        if only_factor==True:#############getDataReady有更新,only_factor==True时，只get因子，不get Return和weight。默认为False。
            data=pd.concat([other_columns_df, self.other_factors[1]], axis=1)
        else:
            if d==0:
                data = pd.concat([self.other_factors[0][timelist_columns], other_columns_df, self.other_factors[1], self.other_factors[2][timelist_columns]], axis=1)
            else:
                data = pd.concat([self.other_factors[0].shift(periods=-d,axis=1)[timelist_columns], other_columns_df, self.other_factors[1], self.other_factors[2][timelist_columns]], axis=1)

        #fill alphas & risk factors' nan / inf / -inf with its mean
        # data.T.replace([np.inf, -np.inf], np.nan)
        # for i in range(1,1+len(alphas)):
        for i in range(1-int(only_factor), 1-int(only_factor) + len(alphas) + 9):
            data.iloc[:,i] = data.iloc[:,i].fillna(data.iloc[:,i].mean())

        data = data.dropna()

        return data  #data的默认顺序为：return，alpha(s), risk factor，industry, weight


    def Orth(x, y): #正交标准化函数，静态函数

        x_array = x.as_matrix(columns=None)
        y_array = y.as_matrix(columns=None)
        ols_resid = sm.OLS(y_array, x_array).fit().resid
        
        return ols_resid

    def regression(x, y, weight=None): #正交标准化函数，静态函数
        
        x_array = x.as_matrix(columns=None)
        y_array = y.as_matrix(columns=None)
            
        if weight is None:
            wls_model = sm.OLS(y_array, x_array)
        else:
            wls_model = sm.WLS(y_array, x_array, weights=list(weight))
            
        return wls_model.fit()

    def findCorrelation(correlation_df, strong_corr_value): #correlation_df是correlation matrix，strong_corr_value是自定义的强相关性的临界值，比如0.8

        start = time.time()
	# 因为correlation matrix是一个中心对称矩阵，我们只需要其中一半，所以我们先把一半变为nan，则后面判断时不会重复
        for i in correlation_df.columns:
            for j in correlation_df[i].index:
                if i>=j:
                    correlation_df.ix[i,j]=np.nan
                    
        strong_corr_pairlist = []
        
        for i in correlation_df.columns:
            strong_corr_alpha = correlation_df[(abs(correlation_df[i])>strong_corr_value)].index.tolist()
            if len(strong_corr_alpha):
                for j in strong_corr_alpha:
                    strong_corr_pairlist.append([j,i])

        print('Find correlation using time: ', time.time()-start)

        return strong_corr_pairlist #output是一个 list，里面有相关性强的alpha pairlist



            
