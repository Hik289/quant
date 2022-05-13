# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 10:50:42 2018

@author: neuron

created for portfolio performance attribution
"""

import numpy as np
import pandas as pd
import base_class
import config
from datetime import timedelta
#import matplotlib.pyplot as plt


class attribution():
    
    def __init__(self,start,end,residual_list,risk_factor_list,weight_list,coef_all_df,residual_df,alpha_list,path):
        print("开始跑了啊")
        
        self.bb = base_class.base_class(risk_factor_list)
        self.timelist = self.bb.timelist
        while start not in self.timelist:
            start=(pd.to_datetime(start)+timedelta(1)).strftime('%Y%m%d')
        while end not in self.timelist:
            end=(pd.to_datetime(end)+timedelta(-1)).strftime('%Y%m%d')
        self.start = start
        self.end = end
        self.residual_list=residual_list
        self.risk_factor_list=risk_factor_list
        self.start_point = self.timelist.index(start)
        self.end_point = self.timelist.index(end)
        self.path=path
        #alpha_list准备
        #在不读csv的时候应该去掉setindex
        self.weight = weight_list.set_index("S_INFO_WINDCODE")
        self.factor_return = coef_all_df
        self.residual = residual_df
        self.alpha_list=alpha_list
        self.attribute = self.attribution()
        self.attribute.to_csv(self.path+"attribute"+self.start+"-"+self.end+".csv")
        #self.plot()
        #返回一个每天收益的拆分
    def attribution(self):
        afc = self.bb.getDataReady(self.timelist[self.start_point],config.predict_period,False,*self.residual_list).iloc[:-1]
        all_factor_columns = self.alpha_list+self.risk_factor_list+afc.columns[11:-1].tolist()
        attribute = pd.DataFrame(index = all_factor_columns,columns = self.timelist[self.start_point:self.end_point-config.predict_period])
        attribute.loc["residual"] = np.nan

        for i in range(self.start_point,self.end_point-config.predict_period):
            data = self.bb.getDataReady(self.timelist[i],config.predict_period,False,*self.residual_list).iloc[:,1:(-1)]
            #data的默认顺序为：alpha(s), risk factor，industry
            data = pd.DataFrame(data=data,index=self.weight[self.timelist[i]].index)
            data.columns=attribute.index[:-1]
            daily_attribute = np.sum(data.T.mul(self.weight[self.timelist[i]]).T,axis=0).mul(self.factor_return[self.timelist[i]].tolist())
            daily_residual = pd.DataFrame(self.weight[self.timelist[i]].fillna(0)).T.dot(self.residual[self.timelist[i]].fillna(0))
            daily_attribute.loc["residual"]=daily_residual.iloc[0]
            attribute[self.timelist[i]]=daily_attribute
        return attribute
    
    def plot(self):
        dfn = pd.read_csv(self.path+"attribute"+self.start+"-"+self.end+".csv").set_index("Unnamed: 0")
        dfn = dfn.apply(lambda x: x+1).cumprod(1).apply(lambda x:x-1).iloc[:,-1]
        dfn=dfn/sum(dfn)
        #dfn.plot(kind="bar",figsize=[50,20],fontsize=40)
        ax=dfn.plot(kind="bar",figsize=[10,4],fontsize=10)
        fig = ax.get_figure()
        fig.savefig(self.path+'fig.png')

            
            
            
            