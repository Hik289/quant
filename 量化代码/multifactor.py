import numpy as np
import config
import pandas as pd
import statsmodels.api as sm
import base_class
import os
from datetime import datetime,timedelta
#import tushare as ts
import time
#import matplotlib.pyplot as plt
#from pyecharts import Grid, Line

#multifactor:与pick_alpha的日期可以不同，但须包含优化的日期。可以不与pick_alpha连起来跑。
class multifactor():

    def __init__(self, start, end, risk_factor_list, strongest_alpha_number, window = config.multifactor_window):  #input:alpha 、factor是list，元素是dataframe
        ###### from base_class
        self.alpha_list=[]
        self.bb = base_class.base_class(risk_factor_list)
        self.risk_factor_list = self.bb.read_risk_factors()
        self.timelist = self.bb.timelist
        while start not in self.timelist:
            start=(pd.to_datetime(start)+timedelta(1)).strftime('%Y%m%d')
        while end not in self.timelist:
            end=(pd.to_datetime(end)+timedelta(-1)).strftime('%Y%m%d')
        self.start_point = self.timelist.index(start) - window
        self.end_point = self.timelist.index(end)
        self.start_date_time = self.timelist[self.start_point]
        self.end_date_time = self.timelist[self.end_point]
        self.alpha_number = len(strongest_alpha_number)
        ttt=time.time()
        for l in strongest_alpha_number:
            temp_alpha=base_class.base_class.read_alphas(l)
            # temp_alpha=temp_alpha.iloc[:,self.start_point:self.end_point+1]
            temp_alpha = temp_alpha.loc[:, self.start_date_time:self.timelist[self.end_point + 1]]
            self.alpha_list.append(temp_alpha)
            del(temp_alpha) ##读进来一个alpha，取一段时间的append到alpha_list里面，再删掉这个alpha
        print('time used for reading alpha:',time.time()-ttt)
        self.strongest_alpha_number=strongest_alpha_number
        ###### self use
        self.window = window
        self.period_timelist = self.timelist[self.start_point:self.end_point] 
        ###### output choice
        self.coef = pd.DataFrame()
        self.coef_real = pd.DataFrame()
        self.coef_all_df = pd.DataFrame()
        self.t_stat_df = pd.DataFrame()
        self.std_error_df = pd.DataFrame() 
        self.residual_df = pd.DataFrame(index = self.bb.index) 
        
        self.IC = pd.Series()
        self.real_sum_df = pd.DataFrame()
        
        ##### execute
        ss=time.time()
        self.orth_alpha()
        print('time used for orth alpha:',time.time()-ss)
        sss=time.time()
        self.Step1and2()
        print('Step1and2 used time:',time.time()-sss)
        #self.Step3and4() #这一步用于看因子体系的好坏，目前未用到
        
        #self.coefplot()

    def drop_extreme_value(self, alpha):  # 针对单个alpha去极值的函数

        start_point = self.timelist.index(self.start_date_time)
        end_point = self.timelist.index(self.end_date_time)

        for j in range(start_point, end_point + 1):
            i = self.timelist[j]
            # if (self.timelist[j] == '20150708'):
            #     print(self.timelist[j])
            a = np.abs(alpha[i] - alpha[i].median())
            diff = a.median()

            ##########need better way to fix inf for alpha114 on 20150708
            if(diff == np.inf):
                aa = a.replace(np.inf, np.nan)
                diff = aa.median()
            ##########need better way to fix inf for alpha114 on 20150708

            if diff == 0:
                diff = alpha[i].mad()
            if np.isnan(diff) == True:
                continue
            maxrange = alpha[i].median() + 4 * diff
            minrange = alpha[i].median() - 4 * diff
            alpha[i] = alpha[i].clip(minrange, maxrange)

        return alpha

    def orth_alpha(self): #对alpha去极值、正交化、标准化
        self.residual_list=[]
        print('orthe alpha')
        for a in range(0,len(self.alpha_list)):
            residual_df=pd.DataFrame(index=self.bb.index, columns=self.period_timelist)
            print(a, ' alpha: ', time.asctime( time.localtime(time.time()) ))
            alpha_standard = self.drop_extreme_value(self.alpha_list[a])
            # alpha_standard = self.alpha_list[a]
            for j in range(self.start_point,self.end_point+1):
                # if(self.timelist[j] == '20150708'):
                #     print(self.timelist[j])
                # print(a, ' alpha: ', j, 'date', time.asctime(time.localtime(time.time())))
                data = self.bb.getDataReady(self.timelist[j],config.predict_period,False,alpha_standard)#顺序：alpha因子、风格因子、行业
                if data.empty == False:
                    residual = base_class.base_class.Orth(data.iloc[:,2:-1],data.iloc[:,1]) #data的默认顺序为：return，alpha(s), risk factor，industry, weight
                    residual_standard = (residual - np.mean(residual))/np.std(residual) #残差标准化
                    residual_series = pd.Series(residual_standard,index=data.index)
                    residual_df[self.timelist[j]] = residual_series
            self.residual_list.append(residual_df)
        if config.multifactor_save_residual_list==True:
            for aa in range(0,len(self.residual_list)):
                self.residual_list[aa].to_csv(config.multifactor_des_path+'/residual%s_tau%s.csv'%(self.strongest_alpha_number[aa],config.predict_period))

    def Step1and2(self): #other_factor顺序：alpha因子、风格因子、Return、weight     

        for i in range(self.start_point,self.end_point-config.predict_period):
            data = self.bb.getDataReady(self.timelist[i],config.predict_period,False,*self.residual_list)#顺序：alpha因子、风格因子、行业
            #data的默认顺序为：return，alpha(s), risk factor，industry, weight          
            #r = pd.DataFrame(data=residual,index=data.index)
            #rd = pd.concat([rd,r],axis=1)

##########以下是step2回归############

            #### t+1对t回归
            if data.empty == False:
                datayy = data.iloc[:,0] #return
                dataxx = data.iloc[:,1:(-1)] # alpha + risk + industry
                result = base_class.base_class.regression(dataxx, datayy, data.iloc[:,-1])

                ### t对t回归
                data_real = pd.concat([self.bb.other_factors[0][self.timelist[i]],data.iloc[:,1:]],axis=1) #替换上一步data的第一列
                data_real = data_real.dropna()
                datay_real = data_real.iloc[:,0]
                datax_real = data_real.iloc[:,1:(-1)]
                result_real = base_class.base_class.regression(datax_real, datay_real, data_real.iloc[:,-1])

                ####提取系数
                self.coef[self.timelist[i]] = result.params[:self.alpha_number]
                self.coef_real[self.timelist[i]] = result_real.params[:self.alpha_number]
                self.coef_all_df[self.timelist[i]] = result.params
                self.t_stat_df[self.timelist[i]] = result.tvalues
                self.std_error_df[self.timelist[i]] = result.bse
                d1 = pd.DataFrame(result.resid,index=data.index,columns=[self.timelist[i]])
                self.residual_df = self.residual_df.join(d1)
        if config.multifactor_save_residual_list==True:
            self.coef_all_df.to_csv(config.multifactor_des_path+'/coef_all_df_tau%s.csv'%config.predict_period)
            self.residual_df.to_csv(config.multifactor_des_path+'/residual_df_tau%s.csv'%config.predict_period)

    def Step3and4(self):#coef, coef_real, residual_list

        for i in range(self.start_point+self.window,self.end_point-config.predict_period):
            predict_sum = 0
            real_sum = 0
            for j in range(0,self.alpha_number):
                
                #是一列Series
                predict_sum += self.residual_list[j][self.timelist[i]]*(self.coef.ix[j,self.timelist[i-self.window+1]:self.timelist[i]].mean())
                real_sum += self.residual_list[j][self.timelist[i]]*self.coef_real.ix[j,self.timelist[i]]
                
            self.IC[self.timelist[i]] = predict_sum.corr(real_sum)#每天有一个IC
            self.real_sum_df[self.timelist[i]] = real_sum #to optimization
        self.ICmean = self.IC.mean()
        self.t = self.IC.mean()/self.IC.std()


    def coefplot(self):
        plotdata = self.coef_all_df.T
        indexlist=list(self.coef_all_df.T.index)
        
        #l = Line(width=1000, height=400)
        l = Line(width=1500, height=600)
        #for i in range(0,len(self.coef_all_df)):
        #    plotdata_column = (plotdata.loc[:,i] + 1).cumprod(0)
        #    l.add('%s'%self.coef_all_df.index[i], indexlist, plotdata_column,
        #          is_fill=False, line_opacity=0.8, is_smooth=True, is_datazoom_show=True,
        #          datazoom_type='both',yaxis_min='dataMin')
        
        #data = self.bb.getDataReady(self.timelist[self.start_point],config.predict_period,False,*self.residual_list)
        #data_columns = data.columns
        #del data
       
        for i in range(0,len(self.strongest_alpha_number)):
            plotdata_column = (plotdata.loc[:,i] + 1).cumprod(0)
            l.add('%s'%self.strongest_alpha_number[i], indexlist, plotdata_column,
                  is_fill=False, line_opacity=0.8, is_smooth=True, is_datazoom_show=True,
                  datazoom_type='both',yaxis_min='dataMin')        
        for i in range(len(self.strongest_alpha_number) , len(self.strongest_alpha_number)+len(self.bb.risk_factor_list)):
            plotdata_column = (plotdata.loc[:,i] + 1).cumprod(0)
            l.add('%s'%self.bb.risk_factor_list[i-len(self.strongest_alpha_number)], indexlist, plotdata_column,
                  is_fill=False, line_opacity=0.8, is_smooth=True, is_datazoom_show=True,
                  datazoom_type='both',yaxis_min='dataMin')        
        for i in range(len(self.strongest_alpha_number)+len(self.bb.risk_factor_list) , len(self.coef_all_df)):
            plotdata_column = (plotdata.loc[:,i] + 1).cumprod(0)
            l.add('%s'%config.industry_name_list[i-len(self.strongest_alpha_number)-len(self.bb.risk_factor_list)], indexlist, plotdata_column,
                  is_fill=False, line_opacity=0.8, is_smooth=True, is_datazoom_show=True,
                  datazoom_type='both',yaxis_min='dataMin')
        #grid = Grid(height=600, width=1200)
        #grid.add(l, grid_top="12%", grid_left="10%")
        #grid.render(path='/Users/yutong/Desktop/atest.html')
        
        l.render(path='/Users/yutong/Desktop/coefplot.html')        


'''

# In[ ]:

#if __name__ == '__main__':
start_date_time = '20090803'
end_date_time = '20180427'
risk_factor_list = ['beta','value','momentum','size','earning_yield','volatility','liquidity','leverage','growth']
risk_path = '/hhh/SVN/risk_factor'
obj = multifactor(start_date_time,end_date_time,risk_factor_list, risk_path,residual_list)


# In[ ]:

#OUTPUT:
obj.IC   #一行
obj.ICmean  #一个数
obj.t  #一个数
obj.coef_all_df.iloc[obj.alpha_number:,:]  #dataframe:to optimization
obj.real_sum_df #dataframe:to optimization
obj.residual_df
'''
