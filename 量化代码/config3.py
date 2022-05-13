
import numpy as np
from numpy import *
import pandas as pd
import os
import time

# dir_path = os.path.dirname(os.path.realpath(__file__))
# risk_path='/Volumes/ProjectOne/SVN/risk_factor'
# alpha_path = '/Volumes/ProjectOne/SVN/alpha'
# summary_path = dir_path + '/hhh/SVN/alpha_summary/1_day'
# draw_picture_path = dir_path + '/hhh/SVN/alpha_test_new/1_day'
# multifactor_des_path='/Users/l/Desktop'#save residual list path
# opt_des_path='/Users/l/Desktop'  ###save统计指标的excel
# W_BM = pd.read_csv(dir_path+'/hhh/W_bm_zz_500.csv',index_col='S_CON_WINDCODE') #benchmark
# other_factor_path = risk_path
"""
dir_path = ''
# W_BM = pd.read_csv(dir_path+'D:/work/cubicalQuant/factorModel/ProjectOne/SVN/W_bm_zz_500.csv',index_col='S_CON_WINDCODE') #benchmark
W_BM = pd.read_csv(dir_path+'D:/work/cubicalQuant/factorModel/ProjectOne/optimization/opt_xinge_0913_2/W_bm_zz_500 .csv',index_col='S_INFO_WINDCODE')*100 #benchmark
risk_path = dir_path + 'D:/work/cubicalQuant/factorModel/ProjectOne/optimization/opt_xinge_0913_2/risk_new'
other_factor_path = risk_path
alpha_path = dir_path + 'D:/work/cubicalQuant/factorModel/ProjectOne/optimization/opt_xinge_0913_2/alpha_new'
output_path = 'D:/work/cubicalQuant/factorModel/ProjectOne/optimization/opt_xinge_0913_2/output_new1'
summary_path = output_path
draw_picture_path = output_path
multifactor_des_path= output_path #save residual list path
opt_des_path= output_path  ###save统计指标的excel
"""
'''
#dir_path ='/Users/chaowang/Desktop/Wuyu/Alpha_Raw'
W_BM = pd.read_csv('/Users/chaowang/Desktop/Wuyu/alpha_picture/risk factors/W_bm_zz_500 .csv',index_col='S_INFO_WINDCODE')*100 #benchmark
risk_path = '/Users/chaowang/Desktop/Wuyu/alpha_picture/risk factors'
other_factor_path = '/Users/chaowang/Desktop/Wuyu/alpha_picture/other_factor'
alpha_path = '/Users/chaowang/Desktop/Wuyu/Alpha_Raw'
output_path = '/Users/chaowang/Desktop/Wuyu/alpha_picture/alpha_picture_all_information'
summary_path = output_path
draw_picture_path = output_path
multifactor_des_path= output_path #save residual list path
opt_des_path= output_path  ###save统计指标的excel
'''
dir_path='/usr/intern/wufei/fun_factor/other_factor'
risk_path ='/usr/intern/wufei/fun_factor/risk factors'
other_factor_path =dir_path
alpha_path='/usr/intern/wufei/fun_factor'
output_path='/usr/intern/wufei/fun_factor/alphapic1'
output_path='/usr/intern/wufei/fun_factor/for_com'
summary_path = output_path
draw_picture_path = output_path
multifactor_des_path= output_path #save residual list path
opt_des_path= output_path  ###save统计指标的excel




industry_name_list = ['交通运输','休闲服务','传媒','公用事业','农林牧渔','化工','医药生物','商业贸易',
                      '国防军工','家用电器','建筑材料','建筑装饰','房地产开发Ⅱ','园区开发Ⅱ',
                      '有色金属','机械设备','汽车', '电子','电气设备','纺织服装','综合','计算机',
                      '轻工制造','通信','采掘','钢铁','银行','保险Ⅱ','证券Ⅱ','多元金融Ⅱ',
                      '食品加工','饮料制造']

# 单因子有效性检验参数
is_test_predict_period = True
predict_period = 2
annual_return_threshold = 0.04
IR_threshold = 1.5
t_mean_abs_threshold = 2.0
t_ratio_threshold = 0.5
alpha_correlation_threshold = 0.8

# 多因子回归参数
multifactor_window = 250
multifactor_save_residual_list=True


#optimization
opt_window = 250
opt_te=0.1
opt_tc=0.000
opt_exposure = 1  # =1是放开风险敞口，=0是不放开
#beta,momentum,size,earning_yield,value,volatility,liquidity,leverage,growth
opt_risk_upperbound=[0.01, 100, 100, 100,  0.01,-0.2, 0.01, 0.01, 0.01]
opt_risk_lowerbound=[-0.01, 0.3, -0.3, 0.5, -0.01, -100, -0.01, -0.01, -0.01]
opt_industry_upperbound=0.05
opt_industry_lowerbound=-0.05
opt_exp_method = 60 #可选：'mv'/60、250等正整数/固定值：0(0.0001)
opt_pkg = 'cpy' # cpy（指cvxpy）, msk(指mosek), cpt(指cvxopt)
opt_method = 'abs'   #可选：'abs'/ 4/3 /2
opt_object_function = 1 # =1是目标函数里包含风险，=0是不包含风险。目前cvxpy可用。
opt_object_lambda = 5 #目标函数风险前的系数
opt_constraint = 0 # =1是限制条件包含风险，=0是限制条件不包含风险。目前cvxpy和cvxopt可用。
opt_cvxpy_solver = 'SCS' # ECOS...
opt_exaggerate_times=10000
weight_upperbound=1.05
weight_lowerbound=0.95