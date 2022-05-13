# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 10:17:33 2018

@author: yutong
"""
import pandas as pd
import multifactor
import pick_alpha
import newoptchng
import attribution


start_date_time = '20170106'
end_date_time = '20170601'
alpha_list =  [137]
risk_factor_list = ['beta','momentum','size','earning_yield','value','volatility','liquidity','leverage','growth']
#obj1先pick出来一些alpha，然后第二步obj2对alpha list做回测，然后get一些参数，并输入到第三部分进行优化
obj1 = pick_alpha.pick_alpha(start_date_time,end_date_time,alpha_list,risk_factor_list)

obj2 = multifactor.multifactor(start_date_time, end_date_time, risk_factor_list,[137])
#obj2 = multifactor.multifactor(start_date_time, end_date_time, risk_factor_list,obj1.strongest_alpha_number)
obj3 = newoptchng.opt('20170112','20170201',risk_factor_list,obj2.residual_list,[-137],obj2.coef_all_df,obj2.residual_df)
# #优化时间需要包含在startday和endday之间，代码运行速度可能比较慢
# obj3 = newoptlgst.opt('20170112','20170418',risk_factor_list,obj2.residual_list,obj1.alpha_sign,obj2.coef_all_df,obj2.residual_df)
df1=obj2.residual_list
df2=obj3.WEIGHT
df3=obj2.coef_all_df
df4=obj2.residual_df
# =============================================================================
# path = "D:/Neuron File/Intern/Optimization/"
# df1.to_csv(path+"df1")
# df2.to_csv(path+"df2")
# df3.to_csv(path+"df3")
# df4.to_csv(path+"df4")
# 
# path = "D:/Neuron File/Intern/Optimization/"
# df1 = pd.read_csv(path+"df1.csv")
# df2 = pd.read_csv(path+"df2.csv")
# df3 = pd.read_csv(path+"df3.csv")
# df4 = pd.read_csv(path+"df4.csv")
# =============================================================================
obj4 = attribution.attribution('20170112','20170201',df1,risk_factor_list,df2,df3,df4)
obj4.attribution.to_csv("/home/intern/optimization/output/attribute.csv")