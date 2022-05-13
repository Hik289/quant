# -- coding: utf-8 --
import pandas as pd
import numpy as np
from alpha_Base import Alpha_Base
from copy import deepcopy as copy
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
import h5py
from scipy import stats
import time
from common_lib import QLLIB
import matplotlib.pyplot as plt
import os
import sys

import statsmodels.api as sm

# reload(sys)
# sys.setdefaultencoding('utf-8')

root_path = os.getcwd()
if len(root_path.split('\\')) > 1:
    while root_path.split('\\')[-1] != 'PRODUCT':
        root_path = os.path.dirname(root_path)
else:
    while root_path.split('/')[-1] != 'PRODUCT':
        root_path = os.path.dirname(root_path)
sys.path.append(root_path)


class Alpha(Alpha_Base):

    def __init__(self, cf):
        Alpha_Base.__init__(self, cf)
        self.h5_path_5min = cf['h5_path_5min']
        self.h5_path_1min = cf['h5_path_1min']
        self.h5_path_tick = cf['h5_path_tick']

    def load_other_data(self):
        # following alpha_sxy_ function may need other data apart from the self-loaded data
        # add psx_alpha.load_other_data() to the run.py after psx_alpha.load_data_from_local()
        # succeed if it shows loaded data from other resources

        # due to the same reason result_path remains the same dirpath

        self.data_path = root_path + r'/workspace_data'
        self.result_path = root_path + r'/workspace_intern/data/advanced_result/'

        self.cap_p = QLLIB.read_from_selflib(data_name='cap_p', \
                                             data_path=self.data_path).reindex(index=self.index, columns=self.columns)

        self.oper_profit_fa = QLLIB.read_from_selflib(data_name='oper_profit_fa', \
                                                      data_path=self.data_path).reindex(index=self.index,
                                                                                        columns=self.columns)
        self.e_fa = QLLIB.read_from_selflib(data_name='e_fa', \
                                            data_path=self.data_path).reindex(index=self.index, columns=self.columns)
        self.net_assets = QLLIB.read_from_selflib(data_name='net_assets', \
                                                  data_path=self.data_path).reindex(index=self.index,
                                                                                    columns=self.columns)
        self.oper_profit_qfa_yoy = QLLIB.read_from_selflib(data_name='oper_profit_qfa_yoy', \
                                                           data_path=self.data_path).reindex(index=self.index,
                                                                                             columns=self.columns)
        self.oper_net_cash_flow = QLLIB.read_from_selflib(data_name='oper_net_cash_flow', \
                                                          data_path=self.data_path).reindex(index=self.index,
                                                                                            columns=self.columns)
        self.tot_assets = QLLIB.read_from_selflib(data_name='tot_assets', \
                                                  data_path=self.data_path).reindex(index=self.index,
                                                                                    columns=self.columns)

        self.PB_p = QLLIB.read_from_selflib(data_name='PB_p', \
                                            data_path=self.data_path).reindex(index=self.index, columns=self.columns)
        self.PE_p = QLLIB.read_from_selflib(data_name='PE_p', \
                                            data_path=self.data_path).reindex(index=self.index, columns=self.columns)
        self.oper_rev_fa = QLLIB.read_from_selflib(data_name='oper_rev_fa', \
                                                   data_path=self.data_path).reindex(index=self.index,
                                                                                     columns=self.columns)
        self.tot_shr = QLLIB.read_from_selflib(data_name='tot_shr', \
                                               data_path=self.data_path).reindex(index=self.index, columns=self.columns)
        self.tot_profit_fa = QLLIB.read_from_selflib(data_name='tot_profit_fa', \
                                                     data_path=self.data_path).reindex(index=self.index,
                                                                                       columns=self.columns)

        self.NET_INFLOW_RATE_VALUE = QLLIB.read_from_selflib(data_name='NET_INFLOW_RATE_VALUE', \
                                                             data_path=self.data_path).reindex(index=self.index,
                                                                                               columns=self.columns)
        self.OPEN_NET_INFLOW_RATE_VOLUME = QLLIB.read_from_selflib(data_name='OPEN_NET_INFLOW_RATE_VOLUME', \
                                                                   data_path=self.data_path).reindex(index=self.index,
                                                                                                     columns=self.columns)
        self.CLOSE_NET_INFLOW_RATE_VOLUME = QLLIB.read_from_selflib(data_name='CLOSE_NET_INFLOW_RATE_VOLUME', \
                                                                    data_path=self.data_path).reindex(index=self.index,
                                                                                                      columns=self.columns)
        self.SELL_VALUE_LARGE_ORDER = QLLIB.read_from_selflib(data_name='SELL_VALUE_LARGE_ORDER', \
                                                              data_path=self.data_path).reindex(index=self.index,
                                                                                                columns=self.columns)
        self.SELL_VALUE_SMALL_ORDER = QLLIB.read_from_selflib(data_name='SELL_VALUE_SMALL_ORDER', \
                                                              data_path=self.data_path).reindex(index=self.index,
                                                                                                columns=self.columns)

        self.BUY_VALUE_LARGE_ORDER = QLLIB.read_from_selflib(data_name='BUY_VALUE_LARGE_ORDER', \
                                                             data_path=self.data_path).reindex(index=self.index,
                                                                                               columns=self.columns)
        self.BUY_VALUE_EXLARGE_ORDER = QLLIB.read_from_selflib(data_name='BUY_VALUE_EXLARGE_ORDER', \
                                                               data_path=self.data_path).reindex(index=self.index,
                                                                                                 columns=self.columns)
        self.BUY_VALUE_SMALL_ORDER = QLLIB.read_from_selflib(data_name='BUY_VALUE_SMALL_ORDER', \
                                                             data_path=self.data_path).reindex(index=self.index,
                                                                                               columns=self.columns)

        self.BUY_VOLUME_LARGE_ORDER = QLLIB.read_from_selflib(data_name='BUY_VOLUME_LARGE_ORDER', \
                                                              data_path=self.data_path).reindex(index=self.index,
                                                                                                columns=self.columns)
        self.BUY_VOLUME_EXLARGE_ORDER = QLLIB.read_from_selflib(data_name='BUY_VOLUME_EXLARGE_ORDER', \
                                                                data_path=self.data_path).reindex(index=self.index,
                                                                                                  columns=self.columns)
        self.BUY_VOLUME_SMALL_ORDER = QLLIB.read_from_selflib(data_name='BUY_VOLUME_SMALL_ORDER', \
                                                              data_path=self.data_path).reindex(index=self.index,
                                                                                                columns=self.columns)

        self.VALUE_DIFF_INSTITUTE_ACT = QLLIB.read_from_selflib(data_name='VALUE_DIFF_INSTITUTE_ACT', \
                                                                data_path=self.data_path).reindex(index=self.index,
                                                                                                  columns=self.columns)

        self.TRADES_COUNT = QLLIB.read_from_selflib(data_name='TRADES_COUNT', \
                                                    data_path=self.data_path).reindex(index=self.index,
                                                                                      columns=self.columns)

        print('load data from other resources')

    def alpha_sxy_026(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_026'
        config['alpha_num'] = 'alpha_sxy_026'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        temp1 = self.amount_p / self.volume_p
        temp2 = self.vwap_p - self.vwap_p.shift(1)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_20 = self.trade_day[axis_now - 20]  # 取前20个交易日日期

            vect_alpha = sxy_cl_str(temp1, temp2, axis_20, di, 0.5) * univers

            m = result.loc[di]
            m[:] = vect_alpha

        return result, config

    def alpha_sxy_027(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_027'
        config['alpha_num'] = 'alpha_sxy_027'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result1 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result2 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result3 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        temp = self.close_p - self.close_p.shift(1)
        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_4 = self.trade_day[axis_now - 4 + 1]
            axis_5 = self.trade_day[axis_now - 5 + 1]

            vect_alpha1 = self_min_str(temp, axis_4, di) * univers
            vect_alpha2 = self_max_str(temp, axis_4, di) * univers
            vect_alpha3 = self_mean_str(temp, axis_5, di) * univers

            m1 = result1.loc[di]
            m1[:] = vect_alpha1
            m2 = result2.loc[di]
            m2[:] = vect_alpha2
            m3 = result3.loc[di]
            m3[:] = vect_alpha3

        result4 = sxy_judge_str(result2.shift(1) - 0., temp, result3)
        result = sxy_judge_str(0. - result1.shift(1), temp, result4)
        return result, config

    def alpha_sxy_028(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_028'
        config['alpha_num'] = 'alpha_sxy_028'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result1 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result2 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        return_p = (self.close_p - self.close_p.shift(1)) / self.close_p

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_250 = self.trade_day[axis_now - 250 + 1]
            axis_20 = self.trade_day[axis_now - 20 + 1]  # 取前10个交易日日期

            signal_vect = self_sum_str(return_p, axis_250, axis_20) * univers
            vect_alpha2 = self_std_str(return_p, axis_20, di) * univers
            vect_alpha1 = QLLIB.map_from_mtx(map_df=self.ind_dict[di], factor_vect=signal_vect, \
                                             retn_type='map', define_central='median') * univers

            m2 = result2.loc[di]
            m2[:] = vect_alpha2
            m1 = result1.loc[di]
            m1[:] = vect_alpha1

        result = result1.rank(axis=1) * (-1.) * result2.rank(axis=1)

        return result, config

    def alpha_sxy_029(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_029'
        config['alpha_num'] = 'alpha_sxy_029'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result1 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result2 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result3 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        return_p = (self.close_p - self.close_p.shift(1)) / self.close_p

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_250 = self.trade_day[axis_now - 250 + 1]  # 取前250个交易日日期
            axis_5 = self.trade_day[axis_now - 5 + 1]

            vect_alpha1 = self_sum_str(return_p, axis_250, di) * univers
            vect_alpha2 = sxy_tscorr(self.close_p.rank(axis=1), self.volume_p.rank(axis=1), axis_5, di)
            vect_alpha3 = QLLIB.map_from_signal(style_vect=vect_alpha1, factor_vect=vect_alpha1, \
                                                retn_type='map', define_central='median', level_num=20) * univers

            m1 = result1.loc[di]
            m1[:] = vect_alpha1
            m2 = result2.loc[di]
            m2[:] = vect_alpha2
            m1 = result1.loc[di]
            m1[:] = vect_alpha1
            m3 = result3.loc[di]
            m3[:] = vect_alpha3

        result = result2.rank(axis=1) * (result3.rank(axis=1)) ** 2 * (-1.)
        return result, config

    def alpha_sxy_030(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_030'
        config['alpha_num'] = 'alpha_sxy_030'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result1 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result2 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result3 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result4 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_10 = self.trade_day[axis_now - 10 + 1]  # 取前10个交易日日期
            axis_5 = self.trade_day[axis_now - 5 + 1]

            vect_alpha1 = sxy_tscorr(self.high_p, self.volume_p, axis_5, di) * univers
            vect_alpha2 = sxy_tscorr(self.low_p, self.volume_p, axis_5, di) * univers
            vect_alpha3 = self_std_str(self.close_p, axis_10, di) * univers
            vect_alpha4 = self_std_str(self.open_p, axis_10, di) * univers

            m1 = result1.loc[di]
            m1[:] = vect_alpha1
            m2 = result2.loc[di]
            m2[:] = vect_alpha2
            m3 = result3.loc[di]
            m3[:] = vect_alpha3
            m4 = result4.loc[di]
            m4[:] = vect_alpha4

        result5 = -1. * (result1 - result1.shift(5)) * result3.rank(axis=1)
        result6 = -1. * (result2 - result2.shift(5)) * result4.rank(axis=1)
        result = sxy_judge_str(self.open_p - self.close_p, result5, result6)
        return result, config

    def alpha_sxy_031(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_031'
        config['alpha_num'] = 'alpha_sxy_031'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result1 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result2 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_10 = self.trade_day[axis_now - 10 + 1]

            vect_alpha1 = sxy_tscorr((self.low_p + self.high_p) / 2, self.volume_p, axis_10, di) * univers
            vect_alpha2 = sxy_tscorr(self.vwap_p, self.volume_p, axis_10, di) * univers

            m1 = result1.loc[di]
            m1[:] = vect_alpha1
            m2 = result2.loc[di]
            m2[:] = vect_alpha2

        return result1.rank(axis=1) / result2.rank(axis=1), config

    def alpha_sxy_032(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_032'
        config['alpha_num'] = 'alpha_sxy_032'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        temp = (self.close_p - self.close_p.shift(1)).shift(1)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_20 = self.trade_day[axis_now - 20 + 1]

            vect_alpha = sxy_EMA(temp, axis_20, di) * univers

            m = result.loc[di]
            m[:] = vect_alpha
        return result, config

    def alpha_sxy_033(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_033'
        config['alpha_num'] = 'alpha_sxy_033'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        temp = (self.close_p - self.close_p.shift(1))

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_20 = self.trade_day[axis_now - 20 + 1]

            vect_alpha = sxy_WMA(temp, axis_20, di, 0.5) * univers

            m = result.loc[di]
            m[:] = vect_alpha
        return result, config

    def alpha_sxy_034(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_034'
        config['alpha_num'] = 'alpha_sxy_034'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_10 = self.trade_day[axis_now - 10 + 1]

            vect_alpha = sxy_WMA(self.high_p - self.low_p, axis_10, di, 0.5) * univers

            m = result.loc[di]
            m[:] = vect_alpha
        result1 = -1. * ((self.high_p - self.low_p) - result).rank(axis=1)
        return result1, config

    def alpha_sxy_035(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_035'
        config['alpha_num'] = 'alpha_sxy_035'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = (self.close_p - self.low_p) * self.open_p / (self.close_p - self.high_p) / self.close_p

        return result, config

    def alpha_sxy_036(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_036'
        config['alpha_num'] = 'alpha_sxy_036'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = (self.close_p - self.low_p) * self.open_p ** 4 / (self.close_p - self.high_p) / self.close_p ** 4

        return result, config

    def alpha_sxy_037(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_037'
        config['alpha_num'] = 'alpha_sxy_037'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result1 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result2 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result3 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        temp = self.close_p - self.close_p.shift(5)
        return_p = self.close_p - self.close_p.shift(1)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_120 = self.trade_day[axis_now - 120 + 1]
            axis_10 = self.trade_day[axis_now - 10 + 1]

            vect_alpha1 = self_tsrank_str(temp, axis_120, di) * univers
            vect_alpha2 = self_sum_str(return_p, axis_120, di) * univers
            vect_alpha3 = self_tsrank_str(self.volume_p, axis_10, di) * univers

            m1 = result1.loc[di]
            m1[:] = vect_alpha1
            m2 = result2.loc[di]
            m2[:] = vect_alpha2
            m3 = result3.loc[di]
            m3[:] = vect_alpha3
        result = (60 - result1) * result2.rank(axis=1) * (5 - result3)
        return result, config

    def alpha_sxy_038(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_038'
        config['alpha_num'] = 'alpha_sxy_038'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result1 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result2 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result3 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result4 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_5 = self.trade_day[axis_now - 5 + 1]

            vect_alpha1 = sxy_tscorr(self.high_p, self.volume_p.rank(axis=1), axis_5, di) * univers
            vect_alpha2 = sxy_tscorr(self.high_p.rank(axis=1), self.volume_p, axis_5, di) * univers
            vect_alpha3 = sxy_tscorr(self.low_p, self.volume_p.rank(axis=1), axis_5, di) * univers
            vect_alpha4 = sxy_tscorr(self.low_p.rank(axis=1), self.volume_p, axis_5, di) * univers

            m1 = result1.loc[di]
            m1[:] = vect_alpha1
            m2 = result2.loc[di]
            m2[:] = vect_alpha2
            m3 = result3.loc[di]
            m3[:] = vect_alpha3
            m4 = result4.loc[di]
            m4[:] = vect_alpha4
        result = -1. * (result1 * result2 * result3 * result4) ** (0.25)
        return result, config

    def alpha_sxy_039(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_039'
        config['alpha_num'] = 'alpha_sxy_039'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result1 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result2 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result3 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result4 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        temp = self.close_p - self.close_p.shift(1)

        for di in self.index:
            axis_now = self.trade_day.index(di)
            axis_20 = self.trade_day[axis_now - 20 + 1]

            vect_alpha1 = self_mean_str(self.close_p, axis_20, di)

            m1 = result1.loc[di]
            m1[:] = vect_alpha1

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_20 = self.trade_day[axis_now - 20 + 1]
            axis_2 = self.trade_day[axis_now - 2 + 1]

            vect_alpha2 = self_mean_str(temp, axis_20, di) * univers
            vect_alpha3 = sxy_tscorr(self.volume_p, self.close_p, axis_2, di) * univers
            vect_alpha4 = sxy_tscorr(result1, self.close_p, axis_2, di) * univers

            m2 = result2.loc[di]
            m2[:] = vect_alpha2
            m3 = result3.loc[di]
            m3[:] = vect_alpha3
            m4 = result4.loc[di]
            m4[:] = vect_alpha4

        result = result2.rank(axis=1) * (-1. * result3) * result4
        return result, config

    def alpha_sxy_040(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_040'
        config['alpha_num'] = 'alpha_sxy_040'
        config['decay_days'] = 5
        config['res_flag'] = 0
        temp1 = (self.close_p - self.close_p.shift(5)) / self.close_p.shift(5) / 5
        temp2 = np.abs(self.close_p - self.close_p.shift(1))
        result = sxy_judge_str(0 - (temp1 - temp1.shift(5)), temp2, -1. * temp2)

        return result * self.univers, config

    def alpha_sxy_041(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_041'
        config['alpha_num'] = 'alpha_sxy_041'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result1 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        temp1 = np.abs(self.close_p - self.close_p.shift(1))

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_10 = self.trade_day[axis_now - 10 + 1]

            vect_alpha1 = self_min_str(temp1, axis_10, di) * univers

            m1 = result1.loc[di]
            m1[:] = vect_alpha1

        temp2 = (self.close_p - self.close_p.shift(5)) / 5
        temp3 = (temp2 - temp2.shift(5))
        result2 = sxy_judge_str(result1 + temp3, -1 * np.abs(temp2), temp1)
        result = sxy_judge_str(result1 - temp3, np.abs(temp2), result2)

        return result, config

    def alpha_sxy_042(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_042'
        config['alpha_num'] = 'alpha_sxy_042'
        config['decay_days'] = 5
        config['res_flag'] = 0

        result = ((self.close_p - self.low_p) - (self.high_p - self.low_p)) / (
                    self.high_p - self.close_p) / self.volume_p

        return result.rank(axis=1) * self.univers, config

    def alpha_sxy_043(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_043'
        config['alpha_num'] = 'alpha_sxy_043'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result1 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result2 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        temp = self.close_p - self.close_p.shift(1)

        map_dict, style_ts = QLLIB.generate_ts_map_info(self.cap_p, self.close_p.reindex(index=self.index),
                                                        define_central='median', level_num=20)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_120 = self.trade_day[axis_now - 120 + 1]

            vect_alpha1 = sxy_tscorr(temp, temp.shift(1), axis_120, di) * univers
            # vect_alpha= ind_neutral(vect_alpha_p, map_dict[di])

            m1 = result1.loc[di]
            m1[:] = vect_alpha1

        result1 = result1 * temp / self.close_p
        for di in self.index:
            vect_alpha2 = ind_neutral(result1.loc[di], map_dict[di])

            m2 = result2.loc[di]
            m2[:] = vect_alpha2

        return result2, config

    def alpha_sxy_044(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_044'
        config['alpha_num'] = 'alpha_sxy_044'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result1 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result2 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result3 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result4 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        for di in self.index:
            univers = self.univers.loc[di]
            ind_dict = self.ind_dict[di].reindex(columns=self.columns)
            vect_alpha4 = ind_neutral(self.vwap_p.loc[di], ind_dict) * univers

            m4 = result4.loc[di]
            m4[:] = vect_alpha4

        for di in self.index:
            axis_now = self.trade_day.index(di)
            axis_4 = self.trade_day[axis_now - 4 + 1]

            vect_alpha1 = sxy_tscorr(result4, self.volume_p, axis_4, di)

            m1 = result1.loc[di]
            m1[:] = vect_alpha1

        for di in self.index:
            axis_now = self.trade_day.index(di)
            axis_8 = self.trade_day[axis_now - 8 + 1]

            vect_alpha2 = sxy_decay_str(result1, axis_8, di)

            m2 = result2.loc[di]
            m2[:] = vect_alpha2

        for di in self.index:
            axis_now = self.trade_day.index(di)
            axis_6 = self.trade_day[axis_now - 6 + 1]

            vect_alpha3 = self_tsrank_str(result2, axis_6, di)

            m3 = result3.loc[di]
            m3[:] = vect_alpha3

        return -1. * result3, config

    def alpha_sxy_045(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_045'
        config['alpha_num'] = 'alpha_sxy_045'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result1 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result2 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        result4 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        for di in self.index:
            univers = self.univers.loc[di]
            ind_dict = self.ind_dict[di].reindex(columns=self.columns)
            vect_alpha4 = ind_neutral(self.vwap_p.loc[di], ind_dict) * univers

            m4 = result4.loc[di]
            m4[:] = vect_alpha4

        for di in self.index:
            axis_now = self.trade_day.index(di)
            axis_4 = self.trade_day[axis_now - 4 + 1]

            vect_alpha1 = sxy_tscorr(result4, self.volume_p, axis_4, di)

            m1 = result1.loc[di]
            m1[:] = vect_alpha1

        for di in self.index:
            axis_now = self.trade_day.index(di)
            axis_8 = self.trade_day[axis_now - 8 + 1]

            vect_alpha2 = sxy_decay_str(result1, axis_8, di)

            m2 = result2.loc[di]
            m2[:] = vect_alpha2

        return -1. * result2, config

    def alpha_sxy_046(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_046'
        config['alpha_num'] = 'alpha_sxy_046'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_20 = self.trade_day[axis_now - 20 + 1]

            vect_alpha = sxy_decay_str(self.high_p + self.low_p - self.vwap_p * 2, axis_20, di) * univers

            m = result.loc[di]
            m[:] = vect_alpha
        return result.rank(axis=1), config

    def alpha_sxy_047(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_047'
        config['alpha_num'] = 'alpha_sxy_047'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result1 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result2 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result3 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result4 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_30 = self.trade_day[axis_now - 30 + 1]

            vect_alpha3 = self_mean_str(self.volume_p, axis_30, di) * univers
            vect_alpha4 = self_mean_str(self.vwap_p, axis_30, di) * univers

            m3 = result3.loc[di]
            m3[:] = vect_alpha3
            m4 = result4.loc[di]
            m4[:] = vect_alpha4

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_3 = self.trade_day[axis_now - 3 + 1]
            axis_120 = self.trade_day[axis_now - 120 + 1]

            vect_alpha1 = sxy_tscorr_str(self.high_p, self.volume_p, axis_3, di) * univers
            vect_alpha2 = sxy_tscorr_str(result3, result4, axis_120, di) * univers

            m1 = result1.loc[di]
            m1[:] = vect_alpha1
            m2 = result2.loc[di]
            m2[:] = vect_alpha2

        result6 = sxy_judge_str(result1 / result2 - 1., result1.rank(axis=1) ** 2 / (-1. * result2).rank(axis=1),
                                -1. * (result1.rank(axis=1) ** 2 / result2.rank(axis=1)))
        result5 = sxy_judge_str(result2 - 0., -1. * (result1.rank(axis=1) ** 2 / result2.rank(axis=1)), result6)
        result = sxy_judge_str(result1 - 0., ((-1. * result1).rank(axis=1)) ** 2 / (-1. * result2).rank(axis=1),
                               result5)
        return result, config

    def alpha_sxy_048(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_048'
        config['alpha_num'] = 'alpha_sxy_048'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_20 = self.trade_day[axis_now - 20 + 1]

            vect_alpha = sxy_tscorr(self.high_p.rank(axis=1), self.PE_p.rank(axis=1), axis_20, di) * univers

            m = result.loc[di]
            m[:] = vect_alpha
        return result.rank(axis=1) * (-1.), config

    def alpha_sxy_049(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_049'
        config['alpha_num'] = 'alpha_sxy_049'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result1 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result2 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_100 = self.trade_day[axis_now - 100 + 1]

            vect_alpha1 = self_mean_str(self.PE_p, axis_100, di) * univers
            vect_alpha2 = self_min_str(self.PE_p, axis_100, di) * univers

            m1 = result1.loc[di]
            m1[:] = vect_alpha1
            m2 = result2.loc[di]
            m2[:] = vect_alpha2

        temp = (result1 - result1.shift(100)) / result1.shift(100)
        result = sxy_judge_str(0.5 - temp, -1. * (self.PE_p - self.PE_p.shift(3)), -1 * (self.PE_p - result2))
        return result, config

    def alpha_sxy_050(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_050'
        config['alpha_num'] = 'alpha_sxy_050'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        return_p = self.close_p - self.close_p.shift(1)
        result1 = sxy_judge_str(return_p - 0, self.PE_p, self.close_p)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_10 = self.trade_day[axis_now - 10 + 1]

            vect_alpha = self_idxmax_str(result1, axis_10, di) * univers

            m = result.loc[di]
            m[:] = vect_alpha
        return result, config

    def alpha_sxy_051(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_051'
        config['alpha_num'] = 'alpha_sxy_051'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_9 = self.trade_day[axis_now - 9 + 1]

            vect_alpha = self_tsrank_str(self.PE_p.rank(axis=1), axis_9, di) * univers

            m = result.loc[di]
            m[:] = vect_alpha
        return -1. * result, config

    def alpha_sxy_052(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_052'
        config['alpha_num'] = 'alpha_sxy_052'
        config['decay_days'] = 5
        config['res_flag'] = 0

        temp1 = (2 * self.close_p - self.low_p - self.high_p) / self.close_p
        temp2 = (self.high_p + self.low_p - 2 * self.open_p) / self.open_p
        result = sxy_judge_str(self.open_p - self.close_p, temp1, temp2)
        return result, config

    def alpha_sxy_053(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_053'
        config['alpha_num'] = 'alpha_sxy_053'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result1 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result2 = pd.DataFrame(index=self.index, columns=self.columns, data=-1.)

        ADV_5 = self.volume_p.rolling(window=5, min_periods=2).mean()
        temp = self.close_p.shift(4)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_10 = self.trade_day[axis_now - 10 + 1]

            vect_alpha = self_tsrank_str(temp, axis_10, di) * univers

            m = result1.loc[di]
            m[:] = vect_alpha
        result = sxy_judge_str(ADV_5 - self.volume_p, -1. * result1 * np.sign(temp), result2)
        return result, config

    def alpha_sxy_054(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_054'
        config['alpha_num'] = 'alpha_sxy_054'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result1 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result2 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        return_p = self.close_p - self.close_p.shift(1)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_5 = self.trade_day[axis_now - 5 + 1]

            vect_alpha1 = self_sum_str(self.open_p, axis_5, di) * univers
            vect_alpha2 = self_sum_str(return_p, axis_5, di) * univers

            m1 = result1.loc[di]
            m1[:] = vect_alpha1
            m2 = result2.loc[di]
            m2[:] = vect_alpha2
        result = (result1 * result2).shift(10) - result1 * result2
        return result, config

    def alpha_sxy_055(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_055'
        config['alpha_num'] = 'alpha_sxy_055'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = (2 * self.close_p - self.low_p - self.high_p) / (self.high_p - self.low_p + 0.0001)
        return result, config

    def alpha_sxy_056(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_056'
        config['alpha_num'] = 'alpha_sxy_056'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_10 = self.trade_day[axis_now - 10 + 1]

            vect_alpha = sxy_tscorr(self.close_p, self.open_p, axis_10, di) * univers

            m = result.loc[di]
            m[:] = vect_alpha
        return result, config

    def alpha_sxy_057(self):
        # alpha 17

        config = {}
        config['alpha_name'] = 'alpha_sxy_057'
        config['alpha_num'] = 'alpha_sxy_057'
        config['decay_days'] = 5
        config['res_flag'] = 0

        temp = np.abs(self.close_p - self.close_p.shift(1)) / self.volume_p ** 0.25

        return temp, config

    def alpha_sxy_058(self):
        # alpha 20

        config = {}
        config['alpha_name'] = 'alpha_sxy_058'
        config['alpha_num'] = 'alpha_sxy_058'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_20 = self.trade_day[axis_now - 20 + 1]

            vect_alpha = self_skew_str(self.close_p, axis_20, di) * univers

            m = result.loc[di]
            m[:] = vect_alpha
        return result - result.shift(2), config

    def alpha_sxy_059(self):
        # alpha 25

        config = {}
        config['alpha_name'] = 'alpha_sxy_059'
        config['alpha_num'] = 'alpha_sxy_059'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        temp = self.amount_p / self.TRADES_COUNT
        return_p = self.close_p - self.close_p.shift(1)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_20 = self.trade_day[axis_now - 20 + 1]

            vect_alpha = sxy_cl_str(temp, return_p, axis_20, di, 1 / 16) * univers

            m = result.loc[di]
            m[:] = vect_alpha
        return result, config

    def alpha_sxy_060(self):
        # alpha19

        config = {}
        config['alpha_name'] = 'alpha_sxy_060'
        config['alpha_num'] = 'alpha_sxy_060'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result1 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result2 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result3 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        large = (self.BUY_VOLUME_LARGE_ORDER + self.BUY_VOLUME_EXLARGE_ORDER) / self.volume_p
        small = self.BUY_VOLUME_SMALL_ORDER / self.volume_p
        close = self.CLOSE_NET_INFLOW_RATE_VOLUME

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_7 = self.trade_day[axis_now - 7 + 1]
            axis_230 = self.trade_day[axis_now - 230 + 1]

            vect_alpha1 = self_sum_str(large, axis_7, di) * univers
            vect_alpha2 = sxy_tscorr_str(close, self.close_p, axis_230, di) * univers

            m1 = result1.loc[di]
            m1[:] = vect_alpha1
            m2 = result2.loc[di]
            m2[:] = vect_alpha2

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_20 = self.trade_day[axis_now - 20 + 1]

            vect_alpha3 = self_mean_str(result2.shift(5), axis_20, di) * univers

            m3 = result3.loc[di]
            m3[:] = vect_alpha3

        result = (result1 / 7 - small) * result3
        return result.rank(axis=1), config


# get the first and last num% and do minus
def sxy_cl_str(df_source, cl, last, now, num):
    temp1 = df_source[cl.loc[last:now] < cl.loc[last:now].quantile(num)].mean()
    temp2 = df_source[cl.loc[last:now] > cl.loc[last:now].quantile(1 - num)].mean()
    return temp2 - temp1


#
def sxy_count_str(df_source, co, last, now, num):
    temp = co.loc[last:now].sum() * num
    df = df_source.T
    df = df[co.loc[last:now].max() > temp].T

    return df.mean()


# compare df_source<0 return1 else return 2
def sxy_judge_str(df_source, return1, return2):
    mtx = pd.DataFrame(index=df_source.index, columns=df_source.columns, data=np.nan)
    mtx[df_source < 0] = return1[df_source < 0]
    mtx[df_source >= 0] = return2[df_source >= 0]
    return mtx


def sxy_equalzero_str(df_source, return1, return2):
    mtx = pd.DataFrame(index=df_source.index, columns=df_source.columns, data=np.nan)
    mtx[df_source == 0] = return1[df_source == 0]
    mtx[df_source != 0] = return2[df_source != 0]
    return mtx


# 行业内rank
def ind_rank(df_source, ind_ref):
    ind_source = ind_ref * df_source.transpose()
    rank_result = ind_source.rank(axis=1)
    max_value = rank_result.max(axis=1)
    result = ((rank_result.T) / max_value).T
    return result.sum()


# delay function
def sxy_delay_str(df_source, last, now):
    if last not in df_source.index:
        return np.nan
    mtx = df_source.loc[last:now]
    return mtx.loc[last]


# delta function
def sxy_delta_str(df_source, last, now):
    if last not in df_source.index:
        return np.nan
    if now not in df_source.index:
        return 0.
    mtx = df_source.loc[last:now]
    return mtx.loc[now] - mtx.loc[last]


# tsmean function
def self_mean_str(df_source, last, now):
    mtx = df_source.loc[last:now]
    return mtx.mean(axis=0)


# tssum function
def self_sum_str(df_source, last, now):
    mtx = df_source.loc[last:now]
    return mtx.sum(axis=0)


# tskurt function
def self_kurt_str(df_source, last, now):
    mtx = df_source.loc[last:now]
    return mtx.kurtosis(axis=0)


# tsskew function
def self_skew_str(df_source, last, now):
    mtx = df_source.loc[last:now]
    return mtx.skew(axis=0)


# tsrank function
def self_tsrank_str(df_source, last, now):
    mtx = df_source.loc[last:now]
    return mtx.rank().loc[now]


# tsmax function
def self_max_str(df_source, last, now):
    mtx = df_source.loc[last:now]
    return mtx.max(axis=0)


# idxmax function numpy
def self_idxmax_str_np(df_source, last, now):
    mtx = df_source.loc[last:now]
    vect = np.argsort(-mtx.values, axis=0)[0, :]
    return pd.Series(vect, index=df_source.columns)


# idxmin function numpy
def self_idxmin_str_np(df_source, last, now):
    mtx = df_source.loc[last:now]
    vect = np.argsort(-mtx.values, axis=0)[-1, :]
    return pd.Series(vect, index=df_source.columns)


# idxmax function
def self_idxmax_str(df_source, last, now):
    mtx = pd.DataFrame(df_source.loc[last:now].values, columns=df_source.columns)
    return mtx.idxmax(axis=0)


# idxmin function
def self_idxmin_str(df_source, last, now):
    mtx = pd.DataFrame(df_source.loc[last:now].values, columns=df_source.columns)
    return mtx.idxmin(axis=0)


# return (x-mean)/std zscore
def z_score_org(vect):
    return (vect - vect.mean()) / vect.std()


# tsmin function
def self_min_str(df_source, last, now):
    mtx = df_source.loc[last:now]
    return mtx.min(axis=0)


# tsstd function
def self_std_str(df_source, last, now):
    mtx = df_source.loc[last:now]
    return mtx.std(axis=0)


# sigmoid function
def self_sigmoid(vect):
    result = 1 / (1 + np.exp(-vect))
    return result


# scale function
def sxy_scale(df_source):
    return df_source / df_source.sum()


# tscorrelation factor function
def sxy_tscorr_str(df_source1, df_source2, last, now):
    if last not in df_source1.index:
        return np.nan
    if last not in df_source2.index:
        return np.nan
    mtx1 = df_source1.loc[last:now].transpose()
    mtx2 = df_source2.loc[last:now].transpose()
    return QLLIB.corr_matrix(mtx1, mtx2, retn_vect=True)


#
def MAD_Outlier(arr):
    arr = arr.astype(float)
    if sum(np.isnan(arr.astype(float))) == len(arr):
        return arr
    median = np.nanmedian(arr)
    MAD = np.nanmedian(np.abs(arr - median))
    arr[arr > median + 6 * 1.4826 * MAD] = median + 6 * 1.4826 * MAD
    arr[arr < median - 6 * 1.4826 * MAD] = median - 6 * 1.4826 * MAD
    return arr


#
def std_vect(vect, level=10):
    med = vect.median()
    err = (vect - med).abs().median()
    up_limite = med + level * err
    down_limite = med - level * err
    vect[vect > up_limite] = up_limite
    vect[vect < down_limite] = down_limite
    result = (vect - vect.mean()) / vect.std()
    return result


#
def std_vect_mtx(df_source, level=6):
    result = pd.DataFrame(index=df_source.index, columns=df_source.columns, data=np.nan)
    for di in df_source.index:
        vect = df_source.loc[di]
        vect = std_vect(vect)
        m = result.loc[di]
        m[:] = vect
    return result


#
def std_vect_mean(vect, level=20):
    med = vect.mean()
    err = (vect - med).abs().std()
    up_limite = med + level * err
    down_limite = med - level * err
    vect[vect > up_limite] = up_limite
    vect[vect < down_limite] = down_limite
    result = (vect - vect.mean()) / vect.std()
    return result


#
def self_rank_mtx(df_source):
    df_rank = df_source.rank(axis=1)
    max_value = df_rank.max(axis=1)
    df_rank = ((df_rank.T) / max_value).T
    return df_rank


# WMA function
def sxy_WMA(df_source, last, now, num=1):
    if last not in df_source.index:
        return np.nan
    new_source = df_source.loc[last:now]
    length = new_source.shape[0]

    weight = copy(new_source)
    weight = weight * 0.0 + 1

    m = pd.Series(index=new_source.index)

    ratio = pd.Series(range(length + 1)[1:])

    m[:] = ratio
    new_source = (new_source.T * (num ** m)).T

    weight = ((weight.T) * (num ** m)).T

    new_source = new_source.sum() / weight.sum()
    return new_source


# EMA function
def sxy_EMA(df_source, last, now):
    if last not in df_source.index:
        return np.nan
    new_source = df_source.loc[last:now]
    length = new_source.shape[0]

    weight = copy(new_source)
    weight = weight * 0.0 + 1

    m = pd.Series(index=new_source.index)

    ratio = pd.Series(range(length + 1)[1:])

    m[:] = ratio
    new_source = (new_source.T * m).T

    weight = ((weight.T) * m).T

    new_source = new_source.sum() / weight.sum()
    return new_source


# decay function same as self_EMA
def sxy_decay_str(df_source, last, now):
    if last not in df_source.index:
        return np.nan
    length = df_source.loc[last:now].shape[0]

    ratio = pd.Series(range(length + 1)[1:])
    new_source = (copy(df_source.loc[last:now]).T)
    m = pd.Series(index=new_source.columns)
    m[:] = ratio
    new_source = new_source * m
    new_source = new_source.sum(axis=1) / (length * (length + 1) * 0.5)
    return new_source.T


# tscovariance function
def sxy_tscorr(df_source_1, df_source_2, last, now):
    if last not in df_source_1.index:
        return np.nan
    if last not in df_source_2.index:
        return np.nan
    new_source_1 = df_source_1.loc[last:now]
    new_source_2 = df_source_2.loc[last:now]
    mult = new_source_1 * new_source_2
    cov = mult.mean(axis=0) - new_source_1.mean(axis=0) * new_source_2.mean(axis=0)
    std = new_source_1.std(axis=0) * new_source_2.std(axis=0) + 0.0001
    return cov / std


# return length day mean/std
def std_solve(df_source, length=20):
    mean_source = copy(df_source)
    std_source = copy(df_source)
    days = len(df_source.index)
    for i in range(days):
        begin = i - length
        if begin < 0:
            begin = 0
        m = mean_source.iloc[i]
        m[:] = df_source.iloc[begin:i].mean()
        n = std_source.iloc[i]
        n[:] = df_source.iloc[begin:i].std()
    return (df_source - mean_source) / std_source


# rank function
def self_rank(df_source):
    df_source = df_source.rank()
    min_value = df_source.min()
    max_value = df_source.max()
    if abs(max_value - min_value) < 0.001:
        print('rank error, the max_value = min_value')
        return df_source * 0
    return (df_source - min_value) / (max_value - min_value)


# nomalize function
def self_normalize(df_source):
    min_value = df_source.min()
    max_value = df_source.max()
    if abs(max_value - min_value) < 0.0000001:
        print('rank error, the max_value = min_value')
        return df_source * 0
    return (df_source - min_value) / (max_value - min_value)


# 与self_rank same
def rank(df_source):
    df_source = df_source.rank()
    min_value = df_source.min()
    max_value = df_source.max()
    if abs(max_value - min_value) < 0.001:
        print('rank error, the max_value = min_value')
        return df_source * 0
    return (df_source - min_value) / (max_value - min_value)


# map_df, ind_dict neutral function
def ind_neutral(df_source, ind_p):
    result = ind_p * df_source.transpose()

    mean = result.mean(axis=1)
    std = result.std(axis=1) + 0.000001
    result = (result.T - mean) / std

    result = (result.T).sum(axis=0)
    result[result == 0] = np.nan
    return result


# same as ind_neutral function
def self_free_neutral(df_source, objt):
    result = objt * df_source.transpose()
    mean = result.mean(axis=1)
    std = result.std(axis=1) + 0.000001
    result = (result.T - mean) / std

    result = (result.T).sum(axis=0)
    result[result == 0] = np.nan
    return result


# same as zscore function
def GetZScoreFactor(factor_df):
    ## use zscore x-mean/std
    zscore_factor_df = factor_df.apply(lambda x: (x - x.mean()) / x.std(), axis=1, raw=False)
    return zscore_factor_df


def self_get_objt(df_source):
    # df_source:
    rank1 = rank(df_source)
    result = pd.DataFrame(index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], columns=df_source.index)
    for i in range(10):
        m = copy(rank1)
        up = 1.0 - 0.1 * i
        down = up - 0.1
        m[m > up] = np.nan
        m[m < down] = np.nan
        m[m > 0.0] = 1.0

        result.iloc[i][:] = m
    return result


#
def ema(arr):
    window = len(arr)
    weight = [1.0 / (window - idx) for idx in range(0, window, 1)]
    weight = np.where(np.isnan(arr), 0.0, weight)
    weight_sum = np.sum(weight)
    weight = weight / weight_sum
    return np.nansum(arr * weight)


def ema_psx(arr):
    window = len(arr)
    weight = [idx + 1 for idx in range(0, window, 1)]
    weight = np.where(np.isnan(arr), 0.0, weight)
    weight_sum = np.sum(weight)
    weight = weight / weight_sum
    return np.nansum(arr * weight)














































