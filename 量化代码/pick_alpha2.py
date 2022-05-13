import warnings
import numpy as np
import config
from imp import reload
reload(config)
#from config import *

import pandas as pd
from datetime import datetime,timedelta

import time
import validity
import base_class
#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

class pick_alpha(base_class.base_class):

    def __init__(self, start, end, alpha_list, risk_factor_list):

        self.alpha_list = alpha_list
        self.risk_factor_list = risk_factor_list
        self.alpha_number = len(alpha_list)
        self.risk_factor_number = len(risk_factor_list)

        start_time = time.time()
        self.obj = base_class.base_class(self.risk_factor_list, self.alpha_list) # 当base class实例化之后，所有的risk factor和other factor（return，weight等）都会一次性读进来
        print('Time used for data reading: ', time.time()-start_time)
        self.timelist = self.obj.timelist
        while start not in self.timelist:
            start=(pd.to_datetime(start)+timedelta(1)).strftime('%Y%m%d')
        while end not in self.timelist:
            end=(pd.to_datetime(end)+timedelta(-1)).strftime('%Y%m%d')
        self.period_timelist = self.timelist[self.timelist.index(start):self.timelist.index(end)] # 此处取测试时间段的timelist，和self.timelist不一样，用于之后的coef的df建立columns
        self.start_date_time = self.timelist[self.timelist.index(start) - config.multifactor_window] # 当不处于跑预测周期状态时，时间自动比start time往前推n天，n以馨格的multifactor_window为准
        self.end_date_time = end
        self.index = self.obj.index # 用于之后的coef的df建立index
	# 接下来依次调用函数，顺序为：回归得到t，f，std -> 算出alpha的指标并筛选出好的alpha -> 画出净值曲线保存 -> 找出相关性大的alpha
        self.get_alphas_params()
        time1 = time.time()
        self.alpha_indicator()
        print('Time used for getting alphas indicators: ', time.time()-time1)
        #self.draw_picture() 
        self.select_alpha_correlation()
        print('good_alpha_number is:', self.good_alpha_number)
        print('very_good_alpha_number is:', self.very_good_alpha_number)
        print('strongest_alpha_number is:', self.strongest_alpha_number)
        print('alpha_sign:',self.alpha_sign)
        
    def drop_extreme_value(self,alpha): # 针对单个alpha去极值的函数
        
        start_point = self.timelist.index(self.start_date_time)
        end_point = self.timelist.index(self.end_date_time)
        
        for j in range(start_point,end_point+1):
            i = self.timelist[j]
            a = np.abs(alpha[i] - alpha[i].median())
            diff = a.median()

            ##########need better way to fix inf for alpha114 on 20150708
            if(diff == np.inf):
                aa = a.replace(np.inf, np.nan)
                diff = aa.median()
            ##########need better way to fix inf for alpha114 on 20150708

            if diff == 0:
                diff = alpha[i].mad()
            if np.isnan(diff)==True:
                continue
            maxrange = alpha[i].median() + 4 * diff
            minrange = alpha[i].median() - 4 * diff
            alpha[i] = alpha[i].clip(minrange,maxrange)
            
        return alpha

    def single_alpha_test(self,alpha): # 此函数对单个alpha进行正交化，残差标准化，回归以及创建对应的t，f，std error的dataframe

        start_point = self.timelist.index(self.start_date_time)
        end_point = self.timelist.index(self.end_date_time)-config.predict_period-1
        self.t_df = pd.DataFrame()
        self.coef_df = pd.DataFrame()
        self.std_error_df = pd.DataFrame()
        #if not config.is_test_predict_period: self.residual_df = pd.DataFrame(index=self.index, columns=self.period_timelist)
        ic = []

        for j in range(start_point,end_point+1):
            data = self.obj.getDataReady(self.timelist[j],config.predict_period,False,alpha) #此处用实例调用base class的getdataready函数
                
            if data.empty == False:
                residual = base_class.base_class.Orth(data.ix[:,2:-1],data.ix[:,1]) #data的默认顺序为：return，alpha(s), risk factor，industry, weight
                # residual_standard = (residual - np.mean(residual))/np.std(residual) #残差标准化
                if len(np.unique(residual)) == 1:
                    residual_standard = residual
                else:
                    residual_standard = (residual - np.mean(residual)) / np.std(residual)  # 残差标准化

                data.ix[:,1] = residual_standard #直接把data里原来的alpha那列替换为标准化之后的残差，以便后面的regression
                self.result = base_class.base_class.regression(data.ix[:,1:-1], data.ix[:,0], data.ix[:,-1])
                #if not config.is_test_predict_period: residual_series = pd.Series(residual_standard,index=data.index)
                #if not config.is_test_predict_period: self.residual_df[self.timelist[j]] = residual_series
                self.t_df[self.timelist[j]] = self.result.tvalues
                self.coef_df[self.timelist[j]] = self.result.params
                self.std_error_df[self.timelist[j]] = self.result.bse
                corr_matrix = np.corrcoef(data.ix[:,1].values,data.ix[:,0].values) #求ic，即残差和return的相关系数
                ic.append(corr_matrix[0,1])
     
        self.ic = np.mean(ic) #每个alpha每天有一个ic，所以求平均
        self.icir = np.mean(ic)/np.std(ic)

    def get_alphas_params(self): # 此函数为执行函数，对所有alpha进行single alpha test，output是所有的t，f，std error list，每个list的长度即为alpha的个数

        self.t_list = []
        self.coef_list = []
        self.std_error_list = []
        self.ic_list = []
        self.icir_list = []
        if not config.is_test_predict_period: self.residual_list = []
        
        for i in range(0,len(self.alpha_list)):
            start = time.time()
            alpha = base_class.base_class.read_alphas(self.alpha_list[i])
            print('Time used for reading alpha_%d: '%self.alpha_list[i], time.time()-start)
            alpha_standard = self.drop_extreme_value(alpha)
            self.single_alpha_test(alpha_standard)
            self.ic_list.append(self.ic)
            self.icir_list.append(self.icir)
            self.t_list.append(self.t_df)
            self.coef_list.append(self.coef_df)
            self.std_error_list.append(self.std_error_df)
            #if not config.is_test_predict_period: self.residual_list.append(self.residual_df)
            print('Time used for getting alpha_%d parameters: '%self.alpha_list[i], time.time()-start)


    def alpha_indicator(self): #此函数功能为：1.算出alpha和风险因子的指标 2.根据风险因子的指标筛选alpha

        self.good_alpha_number = []
        self.very_good_alpha_number = []
        if not config.is_test_predict_period: self.good_alpha_residual_list = []
        if not config.is_test_predict_period: self.very_good_alpha_residual_list = []
        self.alpha_IR_dict = {}
        self.alpha_all_attributes_df = pd.DataFrame()
        self.alpha_sign = []
        for i in range(0,len(self.alpha_list)):
            start = time.time()
            t = self.t_list[i] 
            f = self.coef_list[i]
            sigma = self.std_error_list[i]
            icmean = pd.Series(index=range(0,self.risk_factor_number+1))
            icmean[0] = self.ic_list[i]
            icir = pd.Series(index=range(0,self.risk_factor_number+1))
            icir[0] = self.icir_list[i]
            alpha_stats = validity.validity(self.risk_factor_number+1,f,t,sigma,self.start_date_time,self.end_date_time)
            indicators_name = ['annual_return','annual_volatility','IR','t_mean','t_mean_abs','t_ratio','t_ratio_positive','IC','ICIR']
            factors_name = ['alpha%d'%self.alpha_list[i],*self.risk_factor_list]
            d = [alpha_stats.annual_return(),alpha_stats.annual_volatility(),alpha_stats.IR(),alpha_stats.t_mean(),alpha_stats.t_mean_abs(),alpha_stats.t_ratio(),alpha_stats.t_ratio_positive(),icmean,icir]
            summary = pd.concat(d,axis=1)
            summary.index = factors_name
            summary.columns = indicators_name
            if config.is_test_predict_period: summary.to_csv(config.summary_path + '/summary_alpha%d.csv'%self.alpha_list[i])

            self.select_alpha_indicators(i, summary.ix[0,'annual_return'], summary.ix[0,'IR'], summary.ix[0,'t_mean_abs'], summary.ix[0,'t_ratio']) # 此处调用了class中的另一个函数select_alpha_indicator,即在算出alpha指标的同时去筛选alpha
            self.alpha_IR_dict[self.alpha_list[i]] = summary.ix[0,'IR']
            self.alpha_all_attributes_df['alpha_%d'%self.alpha_list[i]] = summary.ix[0,:]

        self.alpha_all_attributes_df = self.alpha_all_attributes_df.T
        if config.is_test_predict_period: self.alpha_all_attributes_df.to_csv(config.summary_path + '/alpha_all_atrributes_table_%d_day.csv'%config.predict_period)

    def draw_picture(self):

        start = time.time()
        self.all_alphas_f_df = pd.DataFrame(index = self.period_timelist)
        
        for i in range(0,len(self.alpha_list)):
            f = self.coef_list[i]
            alpha_f = f.ix[0,:]
            self.all_alphas_f_df['alpha_%d'%self.alpha_list[i]] = alpha_f
            plotdata = (alpha_f + 1).cumprod(0)
            plotdata.index = pd.to_datetime(plotdata.index)
            plotdata.plot()
            if config.is_test_predict_period: plt.savefig(config.draw_picture_path + '/alpha%d.png'%self.alpha_list[i])
            plt.close('all') #关掉上一个循环的画图，保证一条线一个图

        if config.is_test_predict_period: self.all_alphas_f_df.to_csv(config.summary_path + '/all_alphas_coef.csv')
        print('Time used for creating alphas coef dataframe: ', time.time()-start)
 

    def select_alpha_indicators(self, i, annual_return, IR, t_mean_abs, t_ratio): # 筛选alpha的函数，会被alpha indicator函数调用，标准可在config中修改

        if abs(annual_return) >= config.annual_return_threshold and abs(IR) >= config.IR_threshold:
            self.good_alpha_number.append(self.alpha_list[i])
            if annual_return>0:
                self.alpha_sign.append(self.alpha_list[i])
            elif annual_return<0:
                self.alpha_sign.append(-self.alpha_list[i])
            #if not config.is_test_predict_period: self.good_alpha_residual_list.append(self.residual_list[i])

            if abs(t_mean_abs) >= config.t_mean_abs_threshold and t_ratio >= config.t_ratio_threshold:
                self.very_good_alpha_number.append(self.alpha_list[i])
                #if not config.is_test_predict_period: self.very_good_alpha_residual_list.append(self.residual_list[i])

    def select_alpha_correlation(self): #

        good_alpha_f_df = pd.DataFrame()
        self.strongest_alpha_number = []
        
        for i in self.good_alpha_number:
            good_alpha_f_df[i] = self.coef_list[self.alpha_list.index(i)].ix[0,:]

        strong_corr_pairlist = []
        if len(good_alpha_f_df.columns) > 1: #如果只有一个alpha则不需要test correlation，只有多个alpha时才需要找相关性
            corr_matrix = good_alpha_f_df.corr()  # 得到多个alpha的correlation matrix
            strong_corr_pairlist = base_class.base_class.findCorrelation(corr_matrix,config.alpha_correlation_threshold) # 调用base class中的函数
            print('alphas have strong correlation are: ',strong_corr_pairlist)
            
        self.strongest_alpha_number = self.good_alpha_number.copy() #此处一定要写成值传递
        if len(strong_corr_pairlist):
            for i in strong_corr_pairlist:
                if self.alpha_IR_dict[i[0]] > self.alpha_IR_dict[i[1]]:
                    if i[1] in self.strongest_alpha_number:
                        alpha_index=self.strongest_alpha_number.index(i[1])
                        self.strongest_alpha_number.remove(i[1])
                        self.alpha_sign.remove(self.alpha_sign[alpha_index])
                else:
                    if i[0] in self.strongest_alpha_number:
                        alpha_index=self.strongest_alpha_number.index(i[0])
                        self.strongest_alpha_number.remove(i[0])
                        self.alpha_sign.remove(self.alpha_sign[alpha_index])

        #if not config.is_test_predict_period:                         
        #    self.strongest_alpha_residual_list = []
        #    for i in self.strongest_alpha_number:
        #        self.strongest_alpha_residual_list.append(self.residual_list[self.alpha_list.index(i)])
        
        
if __name__ == '__main__': # 当执行main文件时，此处以下不会被执行，所以可以保留没问题

    start_point = '20090803'
    end_point = '20180427'
    
    print('oh!')

    alpha_list = list(range(1,192))
    for i in [20,23,25,27,30,34,51,55,56,59,*list(range(61,71)),73,87,92,141,149,161,162,165,166,167,171,181,182,183]:
        alpha_list.remove(i)
    alpha_list = [42,72,76,78,83,99]

    print('Testing alphas: ',alpha_list)
    risk_factor_list = ['beta','momentum','size','earning_yield','value','volatility','liquidity','leverage','growth']

    obj = pick_alpha(start_point,end_point,alpha_list,risk_factor_list)




    




    
