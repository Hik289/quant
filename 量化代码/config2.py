
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

W_BM = pd.read_csv('/home/intern/wzy/optimization/data/W_bm_zz_500.csv',index_col='S_CON_WINDCODE') #benchmark
risk_path = '/home/intern/wzy/optimization/data/risk factors/'
other_factor_path = risk_path
alpha_path = '/home/intern/wzy/optimization/data/'
output_path = '/home/intern/wzy/optimization/output/'
summary_path = output_path
draw_picture_path = output_path
multifactor_des_path= output_path #save residual list path
opt_des_path= output_path  ###save统计指标的excel

pick_alpha_des_path = '/home/intern/wzy/optimization/output/'

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

daily_pick_alpha_window = 1

# 多因子回归参数
multifactor_window = 250
multifactor_save_residual_list=False
#coef_window=60

#optimization
opt_ls=1
opt_window = 250
opt_te=0.1
opt_tc=0.003
opt_exposure = 1  # =1是放开风险敞口，=0是不放开
#beta,momentum,size,earning_yield,value,volatility,liquidity,leverage,growth

opt_risk_upperbound=[0.01, 100, 100, 100,  0.01,-0.2, 0.01, 0.01, 0.01]
opt_risk_lowerbound=[-0.01, 0.3, -0.3, 0.5, -0.01, -100, -0.01, -0.01, -0.01]
#opt_risk_upperbound=[0.01, 0.3, 0.3, 0.5,  0.01, 0.2, 0.01, 0.01, 0.01]
#opt_risk_lowerbound=[-0.01, -0.3, -0.3, -0.5, -0.01, -0.2, -0.01, -0.01, -0.01]
opt_industry_upperbound=0.05
opt_industry_lowerbound=-0.05
opt_exp_method = 60 #可选：'mv'/60、250等正整数/固定值：0(0.0001)
opt_pkg = 'cpy' # cpy（指cvxpy）, msk(指mosek), cpt(指cvxopt)
opt_method = 'abs'   #可选：'abs'/ 4/3 /2
opt_object_function = 0 # =1是目标函数里包含风险，=0是不包含风险。目前cvxpy可用。
opt_object_lambda = 5 #目标函数风险前的系数
opt_constraint = 0 # =1是限制条件包含风险，=0是限制条件不包含风险。目前cvxpy和cvxopt可用。
opt_cvxpy_solver = 'SCS' # ECOS...
opt_exaggerate_times=10000
weight_upperbound=1.05
weight_lowerbound=0.95