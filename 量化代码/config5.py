
import numpy as np
from numpy import *
import pandas as pd
import os
import time


dir_path = os.path.dirname(os.path.realpath(__file__))


# 公共参数
risk_path = dir_path + '/risk_factor'
other_factor_path = dir_path + '/risk_factor'
industry_name_list = ['交运设备','交通运输','休闲服务','传媒','信息服务','公用事业','农林牧渔',
        '化工','医药生物','商业贸易','国防军工','家用电器','建筑建材','建筑材料','建筑装饰''房地产',
        '有色金属','机械设备','汽车', '电子','电气设备','纺织服装','综合','计算机','轻工制造','通信',
        '采掘','钢铁','银行','非银金融','食品饮料']

# 单因子有效性检验参数
alpha_path = dir_path + '/alpha'
summary_path = dir_path + '/alpha_summary/5_day'
draw_picture_path = dir_path + '/alpha_test_new/5_day'
predict_period = 5
annual_return_threshold = 0.04
IR_threshold = 1.5
t_mean_abs_threshold = 2.0
t_ratio_threshold = 0.5
alpha_correlation_threshold = 0.8

# 多因子回归参数
multifactor_window = 250

#
opt_window = 0



