
import numpy as np
import pandas as pd

#f,sigma,t are dataframe;
#factor is an object,eg.beta

benchmark=0.03
Rf=0.02
initial=10000000

class validity():

    
    def __init__(self,m,f,t,sigma,start,end):  # m是alpha+风格因子的行数。
        self.f = f.ix[:(m-1),start:end]
        self.sigma = sigma.ix[:(m-1),start:end]
        self.t = t.ix[:(m-1),start:end]
        self.start = start
        self.end = end
        self.m = m

    def sum_return(self):
        return (1+self.f).cumprod(1) 
    
    def annual_return(self):
        return self.f.mean(1)*252
    
    def win_rate(self):
        return self.f[self.f>0].count(1)/self.f.count(1)
    
    def annual_volatility(self):
        return self.f.std(1)*np.sqrt(252)

    def IR(self):
        avg_return = self.f.mean(1)
        return_std = self.f.std(1)
        AnnualRet = avg_return*252
        AnnualVol = return_std*np.sqrt(252)
        IR = AnnualRet/AnnualVol
        return IR
    
    def book_value(self,initial):
        return initial*self.sum_return
    
    def max_drawdown(self):
        count=len(self.sum_return()[self.start:self.end])
        max_DD_list = np.zeros(count)
        for i in range(1,count-1):
            max_DD_list[i] = (self.sum_return()[i] - self.sum_return()[i:count-1].min())/self.sum_return()[i]
        max_DD = max(max_DD_list)
        return max_DD

    def t_ratio(self):
        return self.t[np.abs(self.t)>1.96].count(1)/self.t.shape[1]
    
    def t_ratio_positive(self):
        return self.t[self.t>1.96].count(1)/self.t[np.abs(self.t)>1.96].count(1)
    
    def t_mean(self):
        return self.t.mean(1)
    
    def t_mean_abs(self):
        return self.t.abs().mean(1)
    
    def sharpe_ratio(self):
        avg_return = self.f.mean(1)
        return_std = self.f.std(1)
        AnnualRet = avg_return*252
        AnnualVol = self.sigma.mean()*np.sqrt(252)
        sharpe_ratio = (AnnualRet-Rf) /AnnualVol
        return sharpe_ratio
