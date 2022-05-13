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

        # due to loading pre_calculated alpha to make new alpha result_path remains to be the same dirpath

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

    def alpha_sxy_001(self):
        # STD factor

        config = {}
        config['alpha_name'] = 'alpha_sxy_001'
        config['alpha_num'] = 'alpha_sxy_001'
        config['decay_days'] = 5
        config['res_flag'] = 0

        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        return_p = (self.close_p - self.close_p.shift(1)) / self.close_p

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)

            axis_30 = self.trade_day[axis_now - 30 + 1]

            vect_alpha = self_std_str(return_p, axis_30, di) * univers
            m = result.loc[di]
            m[:] = vect_alpha

        return result.rank(axis=1), config

    def alpha_sxy_002(self):
        # VOLBT factor

        config = {}
        config['alpha_name'] = 'alpha_sxy_002'
        config['alpha_num'] = 'alpha_sxy_002'
        config['decay_days'] = 5
        config['res_flag'] = 0

        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        sz_chg_p = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        sz_chg_p[:] = self.sz_chg

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)

            axis_120 = self.trade_day[axis_now - 120 + 1]

            vect_alpha = sxy_tscorr_str(self.chgRate_p, sz_chg_p, axis_120, di) * univers
            m = result.loc[di]
            m[:] = vect_alpha

        return result.rank(axis=1), config

    def alpha_sxy_003(self):
        # MOMENTUM turnover factor

        config = {}
        config['alpha_name'] = 'alpha_sxy_003'
        config['alpha_num'] = 'alpha_sxy_003'
        config['decay_days'] = 5
        config['res_flag'] = 0

        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        return_p = (self.close_p - self.close_p.shift(1)) / self.close_p

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)

            axis_30 = self.trade_day[axis_now - 30]

            vect_alpha = sxy_delta_str(return_p * self.turnover_p, axis_30, di) * univers
            m = result.loc[di]
            m[:] = vect_alpha

        return result, config

    def alpha_sxy_004(self):
        # MOMENTUM factor

        config = {}
        config['alpha_name'] = 'alpha_sxy_004'
        config['alpha_num'] = 'alpha_sxy_004'
        config['decay_days'] = 5
        config['res_flag'] = 0

        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        return_p = (self.close_p - self.close_p.shift(1)) / self.close_p

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)

            axis_60 = self.trade_day[axis_now - 60]

            vect_alpha = sxy_delta_str(return_p, axis_60, di) * univers
            m = result.loc[di]
            m[:] = vect_alpha

        return result, config

    def alpha_sxy_005(self):
        # reverse factor

        config = {}
        config['alpha_name'] = 'alpha_sxy_005'
        config['alpha_num'] = 'alpha_sxy_005'
        config['decay_days'] = 5
        config['res_flag'] = 0

        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)

            axis_3 = self.trade_day[axis_now - 3]

            vect_alpha = sxy_delta_str(self.vwap_p, axis_3, di) * univers
            m = result.loc[di]
            m[:] = vect_alpha

        return result, config

    def alpha_sxy_006(self):
        # return factor

        config = {}
        config['alpha_name'] = 'alpha_sxy_006'
        config['alpha_num'] = 'alpha_sxy_006'
        config['decay_days'] = 5
        config['res_flag'] = 0

        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)

            axis_1 = self.trade_day[axis_now - 1]

            vect_alpha = sxy_delta_str(self.close_p, axis_1, di) * univers
            m = result.loc[di]
            m[:] = vect_alpha

        return result, config

    def alpha_sxy_007(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_007'
        config['alpha_num'] = 'alpha_sxy_007'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_120 = self.trade_day[axis_now - 120 + 1]

            vect_alpha = sxy_tscorr_str(self.vwap_p, self.volume_p, axis_120, di) * univers

            m = result.loc[di]
            m[:] = vect_alpha
        return result, config

    def alpha_sxy_008(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_008'
        config['alpha_num'] = 'alpha_sxy_008'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result2 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result1 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        return_p = self.close_p - self.close_p.shift(1)
        SI_p = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        SI_p[return_p >= 0] = 1.
        SI_p[return_p < 0] = -1.

        for di in self.index:
            # univers  = self.univers.loc[di]
            axis_now = self.trade_day.index(di)

            axis_1 = self.trade_day[axis_now - 1]
            axis_01 = self.trade_day[axis_now + 1]
            # axis_3= self.trade_day[axis_now- 1]

            vect_alpha1 = sxy_delta_str(return_p * self.turnover_p, axis_1, di)
            vect_alpha2 = sxy_delta_str(return_p * self.turnover_p, di, axis_01)

            m1 = result1.loc[di]
            m1[:] = vect_alpha1
            m2 = result2.loc[di]
            m2[:] = vect_alpha2

        result = return_p * return_p.shift(1) * result1 * return_p * return_p.shift(-1) * result2
        result4 = -1. * result / (self.cap_p) - self.volume_p / (self.cap_p) * self.turnover_p

        return result4 * self.univers, config

    def alpha_sxy_009(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_009'
        config['alpha_num'] = 'alpha_sxy_009'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result2 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result1 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        return_p = self.close_p - self.close_p.shift(1)
        SI_p = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        SI_p[return_p >= 0] = 1.
        SI_p[return_p < 0] = -1.

        for di in self.index:
            # univers  = self.univers.loc[di]
            axis_now = self.trade_day.index(di)

            axis_1 = self.trade_day[axis_now - 1]
            axis_01 = self.trade_day[axis_now + 1]
            # axis_3= self.trade_day[axis_now- 1]

            vect_alpha1 = sxy_delta_str(return_p * self.turnover_p, axis_1, di)
            vect_alpha2 = sxy_delta_str(return_p * self.turnover_p, di, axis_01)

            m1 = result1.loc[di]
            m1[:] = vect_alpha1
            m2 = result2.loc[di]
            m2[:] = vect_alpha2

        result = SI_p * SI_p.shift(1) * result1 + SI_p * SI_p.shift(-1) * result2
        result4 = -1. * result / (self.cap_p) - self.volume_p / (self.cap_p) * self.turnover_p

        return result4 * self.univers, config

    def alpha_sxy_010(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_010'
        config['alpha_num'] = 'alpha_sxy_010'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result2 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result1 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        return_p = self.close_p - self.close_p.shift(1)
        SI_p = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        SI_p[return_p >= 0] = 1
        SI_p[return_p < 0] = -1

        for di in self.index:
            # univers  = self.univers.loc[di]
            axis_now = self.trade_day.index(di)

            axis_1 = self.trade_day[axis_now - 1]
            axis_01 = self.trade_day[axis_now + 1]
            # axis_3= self.trade_day[axis_now- 1]

            vect_alpha1 = sxy_delta_str(return_p * self.turnover_p, axis_1, di)
            vect_alpha2 = sxy_delta_str(return_p * self.turnover_p, di, axis_01)

            m1 = result1.loc[di]
            m1[:] = vect_alpha1
            m2 = result2.loc[di]
            m2[:] = vect_alpha2

        result = SI_p * SI_p.shift(1) * result1 + SI_p * SI_p.shift(-1) * result2
        result3 = -1. * result / (self.cap_p / 10000000.0) - self.volume_p / (self.cap_p / 10000000.0) * self.turnover_p
        result3.fillna(0, inplace=True)

        result4 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_30 = self.trade_day[axis_now - 30 + 1]

            vect_alpha3 = self_sum_str(result3, axis_30, di) * univers

            m = result4.loc[di]
            m[:] = vect_alpha3

        return result4, config

    def alpha_sxy_011(self):
        # alpha012-alpha016 based on alpha011 and alpha011 is also an alpha

        config = {}
        config['alpha_name'] = 'alpha_sxy_011'
        config['alpha_num'] = 'alpha_sxy_011'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result2 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result1 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        return_p = self.close_p - self.close_p.shift(1)

        for di in self.index:
            # univers  = self.univers.loc[di]
            axis_now = self.trade_day.index(di)

            axis_1 = self.trade_day[axis_now - 1]
            axis_01 = self.trade_day[axis_now + 1]
            # axis_3= self.trade_day[axis_now- 1]

            vect_alpha1 = sxy_delta_str(return_p * self.turnover_p, axis_1, di)
            vect_alpha2 = sxy_delta_str(return_p * self.turnover_p, di, axis_01)

            m1 = result1.loc[di]
            m1[:] = vect_alpha1
            m2 = result2.loc[di]
            m2[:] = vect_alpha2

        result = return_p * return_p.shift(1) * result1 + return_p * return_p.shift(-1) * result2
        result3 = -1. * result / (self.cap_p / 10000000.0) - self.volume_p / (self.cap_p / 10000000.0) * self.turnover_p
        result3.fillna(0, inplace=True)
        result4 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        for di in self.index:
            # univers  = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_30 = self.trade_day[axis_now - 30 + 1]

            vect_alpha3 = self_sum_str(result3, axis_30, di)

            m = result4.loc[di]
            m[:] = vect_alpha3

        return result4 * self.univers, config

    def alpha_sxy_012(self):
        # free energy

        config = {}
        config['alpha_name'] = 'alpha_sxy_012'
        config['alpha_num'] = 'alpha_sxy_012'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result1 = QLLIB.read_from_selflib(data_name='raw_alpha_sxy_011', data_path=self.result_path)

        result = np.abs(result1) * self.cap_p / 10000000.0

        return result.rank(axis=1), config

    def alpha_sxy_013(self):
        # pressure

        config = {}
        config['alpha_name'] = 'alpha_sxy_013'
        config['alpha_num'] = 'alpha_sxy_013'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result1 = QLLIB.read_from_selflib(data_name='raw_alpha_sxy_011', data_path=self.result_path)

        result = np.log(np.abs(result1)) / self.volume_p

        return result, config

    def alpha_sxy_014(self):
        # inner energy2

        config = {}
        config['alpha_name'] = 'alpha_sxy_014'
        config['alpha_num'] = 'alpha_sxy_014'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result1 = QLLIB.read_from_selflib(data_name='raw_alpha_sxy_011', data_path=self.result_path)
        result2 = QLLIB.read_from_selflib(data_name='raw_alpha_sxy_013', data_path=self.result_path)

        result = result2 / result1 * (self.cap_p / 10000000.0)

        return result.rank(axis=1), config

    def alpha_sxy_015(self):
        # force

        config = {}
        config['alpha_name'] = 'alpha_sxy_015'
        config['alpha_num'] = 'alpha_sxy_015'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result1 = QLLIB.read_from_selflib(data_name='raw_alpha_sxy_011', data_path=self.result_path)

        result = -np.log(np.abs(result1)) / self.turnover_p

        return result, config

    def alpha_sxy_016(self):
        # entropy

        config = {}
        config['alpha_name'] = 'alpha_sxy_016'
        config['alpha_num'] = 'alpha_sxy_016'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result1 = QLLIB.read_from_selflib(data_name='raw_alpha_sxy_012', data_path=self.result_path)
        result2 = QLLIB.read_from_selflib(data_name='raw_alpha_sxy_015', data_path=self.result_path)
        result = (result2 - result1) / (self.cap_p / 10000000.0)

        return result, config

    def alpha_sxy_017(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_017'
        config['alpha_num'] = 'alpha_sxy_017'
        config['decay_days'] = 5
        config['res_flag'] = 0
        return_p = self.close_p - self.close_p.shift(1)
        alpha_ = 0.82
        lambda_ = 4.5
        new_volume_p = self.volume_p * sxy_judge_str(0. - return_p, return_p ** alpha_, lambda_ * (-return_p) ** alpha_)
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_20 = self.trade_day[axis_now - 20 + 1]

            vect_alpha = self_mean_str(new_volume_p, axis_20, di) * univers

            m = result.loc[di]
            m[:] = vect_alpha

        result = result - result.shift(1)
        return result, config

    def alpha_sxy_018(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_018'
        config['alpha_num'] = 'alpha_sxy_018'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        alpha_ = 0.82
        lambda_ = 4.5
        return_p = self.close_p - self.close_p.shift(1)
        new_volume_p = self.volume_p * sxy_judge_str(0. - return_p, return_p ** alpha_, lambda_ * (-return_p) ** alpha_)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_10 = self.trade_day[axis_now - 10 + 1]

            vect_alpha = sxy_tscorr_str(self.open_p, new_volume_p, axis_10, di) * univers

            m = result.loc[di]
            m[:] = vect_alpha

        return -1. * result, config

    def alpha_sxy_019(self):
        config = {}
        config['alpha_name'] = 'alpha_sxy_019'
        config['alpha_num'] = 'alpha_sxy_019'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        return_p = (self.close_p - self.close_p.shift(1)).reindex(index=self.index).fillna(0)
        result.iloc[1, :] = return_p.iloc[1, :].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        temp = copy(result)

        for di in return_p.iloc[2:, :].index:
            axis_now = self.trade_day.index(di)
            axis_yes = self.trade_day[axis_now - 1]
            temp.loc[di] = return_p.loc[di].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            a = (temp.loc[di] != temp.loc[axis_yes]) & (temp.loc[di] != 0)
            b = (temp.loc[di] == temp.loc[axis_yes]) | (temp.loc[di] == 0)

            result.loc[di][a] = np.sign(temp.loc[di])

            result.loc[di][b] = result.loc[axis_yes] + 1 * np.sign(temp.loc[axis_yes])

        return result * self.univers, config

    def alpha_sxy_020(self):
        config = {}
        config['alpha_name'] = 'alpha_sxy_020'
        config['alpha_num'] = 'alpha_sxy_020'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        return_p = (self.close_p - self.open_p).reindex(index=self.index).fillna(0)
        result.iloc[1, :] = return_p.iloc[1, :].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        temp = copy(result)

        for di in return_p.iloc[2:, :].index:
            axis_now = self.trade_day.index(di)
            axis_yes = self.trade_day[axis_now - 1]
            temp.loc[di] = return_p.loc[di].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            result.loc[di][(temp.loc[di] == temp.loc[axis_yes]) | (temp.loc[di] == 0)] = result.loc[
                                                                                             axis_yes] + 1 * np.sign(
                temp.loc[axis_yes])
            result.loc[di][(temp.loc[di] != temp.loc[axis_yes]) & (temp.loc[di] != 0)] = np.sign(temp.loc[di])

        return result * self.univers, config

    def alpha_sxy_021(self):
        config = {}
        config['alpha_name'] = 'alpha_sxy_021'
        config['alpha_num'] = 'alpha_sxy_021'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        close_p = (self.close_p * self.re_p).rolling(window=5, min_periods=2).mean()
        return_p = (close_p - close_p.shift(1)).reindex(index=self.index).fillna(0)
        for i in range(1, 8):
            result.iloc[i, :] = return_p.iloc[i, :].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        temp = copy(result)

        for di in return_p.iloc[8:, :].index:
            axis_now = self.trade_day.index(di)
            axis_y1 = self.trade_day[axis_now - 1]
            axis_y2 = self.trade_day[axis_now - 2]
            axis_y3 = self.trade_day[axis_now - 3]
            axis_y4 = self.trade_day[axis_now - 4]
            axis_y5 = self.trade_day[axis_now - 5]
            axis_y6 = self.trade_day[axis_now - 6]
            axis_y7 = self.trade_day[axis_now - 7]

            temp.loc[di] = return_p.loc[di].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            b = ((temp.loc[axis_y2] * temp.loc[axis_y3] < 0) | (temp.loc[axis_y2] * temp.loc[axis_y4] < 0) | \
                 (temp.loc[axis_y2] * temp.loc[axis_y5] < 0) | (temp.loc[axis_y2] * temp.loc[axis_y6] < 0) | \
                 (temp.loc[axis_y2] * temp.loc[axis_y7] < 0))
            a = ((temp.loc[axis_y1] * temp.loc[axis_y2] < 0) & b)

            result.loc[di][a] = result.loc[axis_y1] * 0 + 1
            result.loc[di][~ a] = result.loc[axis_y1] + 1

        return result * self.univers, config

    def alpha_sxy_022(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_022'
        config['alpha_num'] = 'alpha_sxy_022'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        temp1 = self.volume_p.diff().rank(axis=1)
        temp2 = self.vwap_p.diff().rank(axis=1)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_7 = self.trade_day[axis_now - 7 + 1]

            vect_alpha = sxy_tscorr_str(temp1, temp2, axis_7, di) * univers

            m = result.loc[di]
            m[:] = vect_alpha

        return (-1.) * result, config

    def alpha_sxy_023(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_023'
        config['alpha_num'] = 'alpha_sxy_023'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result1 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result2 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result3 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        return_p = self.close_p - self.close_p.shift(1)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)

            axis_5 = self.trade_day[axis_now - 5 + 1]

            vect_alpha1 = self_std_str(return_p, axis_5, di)
            m1 = result1.loc[di]
            m1[:] = vect_alpha1

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)

            axis_10 = self.trade_day[axis_now - 10 + 1]

            vect_alpha2 = self_idxmax_str(self.close_p, axis_10, di) * univers
            vect_alpha3 = self_idxmax_str(result1, axis_10, di) * univers
            m2 = result2.loc[di]
            m2[:] = vect_alpha2
            m3 = result3.loc[di]
            m3[:] = vect_alpha3

        result = sxy_judge_str(return_p - 0., result3, result2)
        return result, config

    def alpha_sxy_024(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_024'
        config['alpha_num'] = 'alpha_sxy_024'
        config['decay_days'] = 5
        config['res_flag'] = 0

        vect_alpha = np.abs(self.close_p * self.re_p - self.close_p.shift(1) * self.re_p.shift(1))
        result = vect_alpha * (self.close_p * self.re_p - self.open_p * self.re_p)

        return result.rank(axis=1) * self.univers, config

    def alpha_sxy_025(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_025'
        config['alpha_num'] = 'alpha_sxy_025'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        temp = (self.high_p - self.low_p) - 1

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_20 = self.trade_day[axis_now - 20]

            vect_alpha = sxy_cl_str(temp, self.close_p, axis_20, di, 0.25) * univers

            m = result.loc[di]
            m[:] = vect_alpha

        return result, config

    # get the first and last num% and do minus


def sxy_cl_str(df_source, cl, last, now, num):
    temp1 = df_source[cl.loc[last:now] < cl.loc[last:now].quantile(num)].mean()
    temp2 = df_source[cl.loc[last:now] > cl.loc[last:now].quantile(1 - num)].mean()
    return temp2 - temp1


# compare df_source<0 return1 else return 2
def sxy_judge_str(df_source, return1, return2):
    mtx = pd.DataFrame(index=df_source.index, columns=df_source.columns, data=np.nan)
    mtx[df_source < 0] = return1[df_source < 0]
    mtx[df_source >= 0] = return2[df_source >= 0]
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
def self_EMA(df_source, last, now):
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


























