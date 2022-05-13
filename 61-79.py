# -- coding: utf-8 --
import pandas as pd
import numpy as np
from alpha_Base import Alpha_Base
from copy import deepcopy as copy
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
from copy import deepcopy as copy
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

        # we need to get precaluculated csvlog but not alpha log from result_path, so it's different from
        # the previous version

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

    def alpha_sxy_061(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_061'
        config['alpha_num'] = 'alpha_sxy_061'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result1 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        return_p = ((self.close_p - self.close_p.shift(1)) / self.close_p.shift(1)).reindex(index=self.index,
                                                                                            columns=self.columns)

        map_dict1, style_ts = QLLIB.generate_ts_map_info(self.cap_p, return_p, define_central='median', level_num=20)

        for di in self.index:
            univers = self.univers.loc[di]

            vect_alpha1 = ind_neutral(self.PB_p.loc[di], map_dict1[di]) * univers

            m1 = result1.loc[di]
            m1[:] = vect_alpha1

        return result1.rank(axis=1), config

    def alpha_sxy_062(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_062'
        config['alpha_num'] = 'alpha_sxy_062'
        config['decay_days'] = 5
        config['res_flag'] = 0

        result2 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        return_p = ((self.close_p - self.close_p.shift(1)) / self.close_p.shift(1)).reindex(index=self.index,
                                                                                            columns=self.columns)

        map_dict1, style_ts = QLLIB.generate_ts_map_info(self.cap_p, return_p, define_central='median', level_num=20)

        for di in self.index:
            univers = self.univers.loc[di]

            vect_alpha2 = ind_neutral(self.PE_p.loc[di], map_dict1[di]) * univers

            m2 = result2.loc[di]
            m2[:] = vect_alpha2

        return result2.rank(axis=1), config

    def alpha_sxy_063(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_063'
        config['alpha_num'] = 'alpha_sxy_063'
        config['decay_days'] = 5
        config['res_flag'] = 0

        result3 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        for di in self.index:
            univers = self.univers.loc[di]
            ind_dict = self.ind_dict[di].reindex(columns=self.columns)

            vect_alpha3 = ind_neutral(self.PB_p.loc[di], ind_dict) * univers

            m3 = result3.loc[di]
            m3[:] = vect_alpha3

        return result3.rank(axis=1), config

    def alpha_sxy_064(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_064'
        config['alpha_num'] = 'alpha_sxy_064'
        config['decay_days'] = 5
        config['res_flag'] = 0

        result4 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        for di in self.index:
            univers = self.univers.loc[di]
            ind_dict = self.ind_dict[di].reindex(columns=self.columns)

            vect_alpha4 = ind_neutral(self.PE_p.loc[di], ind_dict) * univers

            m4 = result4.loc[di]
            m4[:] = vect_alpha4

        return result4.rank(axis=1), config

    def alpha_sxy_065(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_065'
        config['alpha_num'] = 'alpha_sxy_065'
        config['decay_days'] = 5
        config['res_flag'] = 0

        result5 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        return_p = ((self.close_p - self.close_p.shift(1)) / self.close_p.shift(1)).reindex(index=self.index,
                                                                                            columns=self.columns)

        map_dict2, style_ts = QLLIB.generate_ts_map_info(self.PE_p, return_p, define_central='median', level_num=20)

        for di in self.index:
            univers = self.univers.loc[di]

            vect_alpha5 = ind_neutral(return_p.loc[di], map_dict2[di]) * univers

            m5 = result5.loc[di]
            m5[:] = vect_alpha5

        return result5, config

    def alpha_sxy_066(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_066'
        config['alpha_num'] = 'alpha_sxy_066'
        config['decay_days'] = 5
        config['res_flag'] = 0

        result6 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        return_p = ((self.close_p - self.close_p.shift(1)) / self.close_p.shift(1)).reindex(index=self.index,
                                                                                            columns=self.columns)

        map_dict3, style_ts = QLLIB.generate_ts_map_info(self.PB_p, return_p, define_central='median', level_num=20)

        for di in self.index:
            univers = self.univers.loc[di]

            vect_alpha6 = ind_neutral(return_p.loc[di], map_dict3[di]) * univers

            m6 = result6.loc[di]
            m6[:] = vect_alpha6
        return result6, config

    def alpha_sxy_067(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_067'
        config['alpha_num'] = 'alpha_sxy_067'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result1 = 1. / self.PB_p

        result1 = result1.rank(axis=1)

        return result1, config

    def alpha_sxy_068(self):
        # alpha 23 24

        config = {}
        config['alpha_name'] = 'alpha_sxy_068'
        config['alpha_num'] = 'alpha_sxy_068'
        config['decay_days'] = 5
        config['res_flag'] = 0

        return_p = self.close_p - self.close_p.shift(1)
        temp1 = self.PE_p - self.PE_p.shift(60)

        result1 = sxy_equalzero_str(temp1, self.PB_p * (return_p - return_p.shift(30)),
                                    temp1 * (return_p - return_p.shift(60)))

        return result1, config

    def alpha_sxy_069(self):
        # alpha 23 24

        config = {}
        config['alpha_name'] = 'alpha_sxy_069'
        config['alpha_num'] = 'alpha_sxy_069'
        config['decay_days'] = 5
        config['res_flag'] = 0

        return_p = self.close_p - self.close_p.shift(1)

        temp2 = self.PB_p - self.PB_p.shift(60)

        result2 = sxy_equalzero_str(temp2, self.PB_p * (return_p - return_p.shift(30)),
                                    temp2 * (return_p - return_p.shift(60)))

        return result2, config

    def alpha_sxy_070(self):
        # alpha 13

        config = {}
        config['alpha_name'] = 'alpha_sxy_070'
        config['alpha_num'] = 'alpha_sxy_070'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        wap = self.vwap_p - self.vwap_p.shift(1)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            # axis_yes = self.trade_day[axis_now- 1]
            axis_60 = self.trade_day[axis_now - 60 + 1]
            # axis_61  = self.trade_day[axis_now-61+ 1]

            Y = (self.turnover_p.shift(1).loc[axis_60:di] * wap.loc[axis_60:di])
            X = self.vwap_p.shift(1).loc[axis_60:di]
            slope = ((X * Y).mean() - X.mean() * Y.mean()) / ((X ** 2).mean() - X.mean() ** 2)
            intercept = Y.mean() - slope * X.mean()

            predictions = slope * (self.vwap_p.loc[di]) + intercept

            vect_alpha = predictions * univers

            m = result.loc[di]
            m[:] = vect_alpha
        return result, config

    def alpha_sxy_071(self):
        # alpha 15

        config = {}
        config['alpha_name'] = 'alpha_sxy_071'
        config['alpha_num'] = 'alpha_sxy_071'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result1 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        for di in self.index:
            univers = self.univers.loc[di]

            vect_alpha1 = QLLIB.map_from_mtx(map_df=self.ind_dict[di], factor_vect=self.PB_p.loc[di], \
                                             retn_type='map', define_central='median') * univers

            m1 = result1.loc[di]
            m1[:] = vect_alpha1

        result1 = self.PE_p / result1

        return result1.rank(axis=1), config

    def alpha_sxy_072(self):
        # alpha 15

        config = {}
        config['alpha_name'] = 'alpha_sxy_072'
        config['alpha_num'] = 'alpha_sxy_072'
        config['decay_days'] = 5
        config['res_flag'] = 0

        result2 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        for di in self.index:
            univers = self.univers.loc[di]

            vect_alpha2 = QLLIB.map_from_mtx(map_df=self.ind_dict[di], factor_vect=self.PE_p.loc[di], \
                                             retn_type='map', define_central='median') * univers

            m2 = result2.loc[di]
            m2[:] = vect_alpha2

        result2 = self.PB_p / result2

        return result2.rank(axis=1), config

    def alpha_sxy_073(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_073'
        config['alpha_num'] = 'alpha_sxy_073'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        result1 = pd.DataFrame(index=self.index, columns=self.columns, data=1)
        result0 = pd.DataFrame(index=self.index, columns=self.columns, data=0)

        return_p = (self.close_p - self.close_p.shift(1)).fillna(0)
        return_1 = sxy_judge_str(return_p - 0., -result1, result0).rolling(window=7, min_periods=1).sum()
        return_2 = sxy_judge_str(0. - return_p, result1, result0).rolling(window=7, min_periods=1).sum()
        result = return_1 + return_2

        return result, config

    def alpha_sxy_074(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_074'
        config['alpha_num'] = 'alpha_sxy_074'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        data_path_ = self.result_path
        wavelet = QLLIB.read_from_selflib(data_name='sxy_wavelet', data_path=data_path_).fillna(0)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_10 = self.trade_day[axis_now - 10 + 1]

            vect_alpha = self_std_str(wavelet, axis_10, di) * univers

            m = result.loc[di]
            m[:] = vect_alpha
        return -1. * result, config

    def alpha_sxy_075(self):
        # alpha075- alpha079 based on the data_update module

        config = {}
        config['alpha_name'] = 'alpha_sxy_075'
        config['alpha_num'] = 'alpha_sxy_075'
        config['decay_days'] = 5
        config['res_flag'] = 0
        data_path_ = self.result_path
        return_wavelet = QLLIB.read_from_selflib(data_name='sxy_wavelet_return', data_path=data_path_).fillna(0)

        return_last_wavelet = QLLIB.read_from_selflib(data_name='sxy_wavelet_last_return', data_path=data_path_).fillna(
            0)

        result = return_wavelet - return_last_wavelet
        return result, config

    def alpha_sxy_076(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_076'
        config['alpha_num'] = 'alpha_sxy_076'
        config['decay_days'] = 5
        config['res_flag'] = 0
        data_path_ = self.result_path

        result4 = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        large = (self.BUY_VOLUME_LARGE_ORDER + self.BUY_VALUE_EXLARGE_ORDER) / self.volume_p
        inst = self.VALUE_DIFF_INSTITUTE_ACT

        for di in self.index:
            # univers  = self.univers.loc[di]
            axis_now = self.trade_day.index(di)

            axis_60 = self.trade_day[axis_now - 60 + 1]

            vect_alpha4 = self_mean_str(large, axis_60, di)
            m4 = result4.loc[di]
            m4[:] = vect_alpha4

        result4 = large - result4
        result1 = QLLIB.read_from_selflib(data_name='sxy_wavelet_Areturn_rank', data_path=data_path_).fillna(0)
        result2 = QLLIB.read_from_selflib(data_name='sxy_wavelet_return_mean', data_path=data_path_).fillna(0)

        result3 = 2 * np.sign(inst) + np.sign(result4)

        result = sxy_equalzero_str(result3, result1, result2) * self.univers
        return result, config

    def alpha_sxy_077(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_077'
        config['alpha_num'] = 'alpha_sxy_077'
        config['decay_days'] = 5
        config['res_flag'] = 0

        data_path_ = self.result_path

        close_wavelet = QLLIB.read_from_selflib(data_name='sxy_wavelet_close', data_path=data_path_).fillna(0)
        result = (close_wavelet - self.close_p) / close_wavelet

        return result, config

    def alpha_sxy_078(self):
        config = {}
        config['alpha_name'] = 'alpha_sxy_078'
        config['alpha_num'] = 'alpha_sxy_078'
        config['decay_days'] = 5
        config['res_flag'] = 0

        data_path_ = self.result_path
        vwap_mean_wavelet = QLLIB.read_from_selflib(data_name='sxy_wavelet_vwap_mean', data_path=data_path_).fillna(0)

        volume_wavelet = QLLIB.read_from_selflib(data_name='sxy_wavelet_volume', data_path=data_path_).fillna(0)
        wavelet = QLLIB.read_from_selflib(data_name='sxy_wavelet', data_path=data_path_).fillna(0)

        result = vwap_mean_wavelet * volume_wavelet / wavelet
        return result, config

    def alpha_sxy_079(self):

        config = {}
        config['alpha_name'] = 'alpha_sxy_079'
        config['alpha_num'] = 'alpha_sxy_079'
        config['decay_days'] = 5
        config['res_flag'] = 0
        data_path_ = self.result_path
        close_wavelet = QLLIB.read_from_selflib(data_name='sxy_wavelet_close', data_path=data_path_).fillna(0)

        close_last_wavelet = QLLIB.read_from_selflib(data_name='sxy_wavelet_last_close', data_path=data_path_).fillna(0)

        result = close_wavelet - close_last_wavelet
        return result, config

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















































