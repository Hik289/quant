# -- coding: utf-8 --
from alpha_Base import Alpha_Base
from datetime import datetime
# from prodlib import eventlib
import pandas as pd
import numpy as np
import time
import os
from copy import deepcopy as copy
import sys

time0 = time.time()

import h5py

root_path = os.getcwd()
if len(root_path.split('\\')) > 1:
    while root_path.split('\\')[-1] != 'PRODUCT':
        root_path = os.path.dirname(root_path)
else:
    while root_path.split('/')[-1] != 'PRODUCT':
        root_path = os.path.dirname(root_path)
sys.path.append(root_path)

from common_lib import QLLIB

# data_path     = root_path+r'/workspace_data'
data_path = root_path + r'/workspace_data_longterm'
result_path = root_path + r'/workspace_intern/data/alpha_result'
advanced_path = root_path + r'/workspace_intern/data/advanced_result'

################### 暂时用不上 #########################
today_temp = root_path + r'/workspace_intern/data/today_temp'
split_path = root_path + r'/workspace_intern/data/split_factor'
intraday_path = root_path + r'/local_data/intraday_result'
h5_path_5min = root_path + r'/local_data/5min_tmp'
h5_path_1min = root_path + r'/local_data/1min_tmp'
h5_path_tick = root_path + r'/local_data/tick_repaire'

backfill_date = '20170103'
end_date = '20200730'

config = {'backfill_date': backfill_date,
          'end_date': end_date,

          'alpha_name': 'test',

          'fast_count_flag': False,
          'data_path': data_path,
          'result_path': result_path,
          'data_window': 41,
          'decay_days': 41,
          'intraday_path': intraday_path,
          'advanced_path': advanced_path,
          'h5_path_5min': h5_path_5min,
          'h5_path_1min': h5_path_1min,
          'h5_path_tick': h5_path_tick,
          'today_temp': today_temp,
          'advanced_treat_flag': 0,
          'save_org_flag': 0,
          'save_today_flag': 0,
          'suspend_flag': 1,
          'updown_flag': 0,
          'saveResult': 1,
          'write_to_db': 0,
          'repair_flag': 0,
          'system': 'linux',  # windows linux
          'save_intraday_flag': 0,
          'h5_path_1min': h5_path_1min,
          'h5_path_5min': h5_path_5min,
          'data_source': ('volume_price', 'style_all', 'index'),

          'univers': 3500}


class Alpha(Alpha_Base):

    def __init__(self, cf):
        Alpha_Base.__init__(self, cf)
        self.h5_path_5min = cf['h5_path_5min']
        self.h5_path_1min = cf['h5_path_1min']
        self.h5_path_tick = cf['h5_path_tick']
        self.result_path = root_path + r'/workspace_intern/data/alpha_result'

        # put the tempdata into workspaceintern/data/alpha_result
        # this part is for alpha75-alpha79

    def wavelet_v1(self):
        # wavelt_v1: mark the distance of the turnpoint of stockline

        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        return_p = (self.close_p - self.close_p.shift(1)).reindex(index=self.index).fillna(0)
        result.iloc[0, :] = return_p.iloc[1, :].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        temp = copy(result)

        for di in return_p.iloc[1:, :].index:
            axis_now = self.trade_day.index(di)
            axis_y1 = self.trade_day[axis_now - 1]

            temp.loc[di] = return_p.loc[di].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

            a = ((temp.loc[axis_y1] * temp.loc[di] < 0))

            result.loc[di][a] = result.loc[axis_y1] * 0 + 1
            result.loc[di][~ a] = result.loc[axis_y1] + 1

        result[result < 0] = - result
        result = (result * self.univers).fillna(0)
        result.to_csv(self.result_path + '/wavelet_v1.csv')
        return result

    def wavelet_v2(self):
        # wavelet_v2: mark the distance of little wavelet of the stockline

        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        return_p = (self.close_p - self.close_p.shift(1)).reindex(index=self.index).fillna(0)
        for i in range(0, 5):
            result.iloc[i, :] = return_p.iloc[i, :].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        temp = copy(result)

        for di in return_p.iloc[5:, :].index:
            axis_now = self.trade_day.index(di)
            axis_y1 = self.trade_day[axis_now - 1]
            axis_y2 = self.trade_day[axis_now - 2]
            axis_y3 = self.trade_day[axis_now - 3]
            axis_y4 = self.trade_day[axis_now - 4]
            axis_y5 = self.trade_day[axis_now - 5]

            temp.loc[di] = return_p.loc[di].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            b = ((temp.loc[axis_y2] * temp.loc[axis_y3] < 0) | \
                 (temp.loc[axis_y2] * temp.loc[axis_y4] < 0) | (temp.loc[axis_y2] * temp.loc[axis_y5] < 0))
            a = ((temp.loc[axis_y1] * temp.loc[axis_y2] < 0) & b)

            result.loc[di][a] = result.loc[axis_y1] * 0 + 1
            result.loc[di][~ a] = result.loc[axis_y1] + 1

        result[result < 0] = - result
        result = (result * self.univers).fillna(0)
        result.to_csv(self.result_path + '/wavelet_v2.csv')
        return result

    def wavelet_v3(self):
        # wavelet_v3: mark the distance of the little wavelet of the 5-day AVGline

        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        close_p = (self.close_p * self.re_p).rolling(window=5, min_periods=2).mean()
        return_p = (close_p - close_p.shift(1)).reindex(index=self.index).fillna(0)
        for i in range(0, 7):
            result.iloc[i, :] = return_p.iloc[i, :].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        temp = copy(result)

        for di in return_p.iloc[7:, :].index:
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

        result[result < 0] = - result
        result = (result * self.univers).fillna(0)
        result.to_csv(self.result_path + '/wavelet_v3.csv')
        return result

    # temp0- temp4 write down the tempdata based on the v1- v3 wavelet markpoint

    def sxy_temp0(self, wavelet_name):
        wavelet = QLLIB.read_from_selflib(data_name=wavelet_name, data_path=self.result_path)

        wavelet.to_csv(self.result_path + '/sxy_wavelet.csv')

        last_wavelet = self.sxy_wavelet_delta_str(wavelet, wavelet) + wavelet

        last_wavelet.to_csv(self.result_path + '/sxy_last_wavelet.csv')

        return

    def sxy_temp1(self, wavelet_name):
        wavelet = QLLIB.read_from_selflib(data_name=wavelet_name, data_path=self.result_path)

        return_p = self.close_p - self.close_p.shift(1)

        return_wavelet = self.sxy_wavelet_delta_str(return_p, wavelet)
        return_wavelet.to_csv(self.result_path + '/sxy_wavelet_return.csv')

        last_wavelet = QLLIB.read_from_selflib(data_name='sxy_last_wavelet', data_path=self.result_path)

        return_last_wavelet = self.sxy_wavelet_delta_str(return_p, last_wavelet)
        return_last_wavelet.to_csv(self.result_path + '/sxy_wavelet_last_return.csv')

        return

    def sxy_temp2(self, wavelet_name):
        wavelet = QLLIB.read_from_selflib(data_name=wavelet_name, data_path=self.result_path)
        return_p = self.close_p - self.close_p.shift(1)

        wavelet_Areturn_rank = self.sxy_wavelet_rank_str(return_p * wavelet, wavelet)
        wavelet_Areturn_rank.to_csv(self.result_path + '/sxy_wavelet_Areturn_rank.csv')

        wavelet_return_mean = self.sxy_wavelet_mean_str(return_p, wavelet)
        wavelet_return_mean.to_csv(self.result_path + '/sxy_wavelet_return_mean.csv')

        return

    def sxy_temp3(self, wavelet_name):
        wavelet = QLLIB.read_from_selflib(data_name=wavelet_name, data_path=self.result_path)
        close_wavelet = self.sxy_wavelet_delta_str(self.close_p, wavelet)

        close_wavelet.to_csv(self.result_path + '/sxy_wavelet_close.csv')

        last_wavelet = QLLIB.read_from_selflib(data_name='sxy_last_wavelet', data_path=self.result_path)

        close_last_wavelet = self.sxy_wavelet_delta_str(self.close_p, last_wavelet)

        close_last_wavelet.to_csv(self.result_path + '/sxy_wavelet_last_close.csv')
        return

    def sxy_temp4(self, wavelet_name):
        wavelet = QLLIB.read_from_selflib(data_name=wavelet_name, data_path=self.result_path)

        volume_wavelet = self.sxy_wavelet_delta_str(self.volume_p, wavelet)
        volume_wavelet.to_csv(self.result_path + '/sxy_wavelet_volume.csv')

        vwap_mean_wavelet = self.sxy_wavelet_mean_str(self.volume_p, wavelet)
        vwap_mean_wavelet.to_csv(self.result_path + '/sxy_wavelet_vwap_mean.csv')

        return

    def before(self, di, num=1):
        axis_now = self.trade_day.index(di)
        return self.trade_day[axis_now - num]

    def sxy_wavelet_delta_str(self, df_source1, df_source2):
        df_source = df_source1.reindex(index=df_source2.index, columns=df_source2.columns)
        result = copy(df_source)
        for di in df_source.index:
            for dj in df_source.columns:
                move = df_source2.loc[di, dj].astype('int')
                day = self.before(di, num=move)

                if day in df_source.index:
                    result.loc[di, dj] = df_source.loc[day, dj]
        return result

    def sxy_wavelet_mean_str(self, df_source1, df_source2):
        df_source = df_source1.reindex(index=df_source2.index, columns=df_source2.columns)
        result = copy(df_source)
        for di in df_source.index:
            for dj in df_source.columns:
                move = df_source2.loc[di, dj].astype('int')
                day = self.before(di, num=move)
                if day in df_source.index:
                    result.loc[di, dj] = df_source.loc[day:di, dj].mean()
        return result

    def sxy_wavelet_max_str(self, df_source1, df_source2):
        df_source = df_source1.reindex(index=df_source2.index, columns=df_source2.columns)
        result = copy(df_source)
        for di in df_source.index:
            for dj in df_source.columns:
                move = df_source2.loc[di, dj].astype('int')
                day = self.before(di, num=move)
                if day in df_source.index:
                    result.loc[di, dj] = df_source.loc[day:di, dj].max()
        return result

    def sxy_wavelet_min_str(self, df_source1, df_source2):
        df_source = df_source1.reindex(index=df_source2.index, columns=df_source2.columns)
        result = copy(df_source)
        for di in df_source.index:
            for dj in df_source.columns:
                move = df_source2.loc[di, dj].astype('int')
                day = self.before(di, num=move)
                if day in df_source.index:
                    result.loc[di, dj] = df_source.loc[day:di, dj].max()
        return result

    def sxy_wavelet_rank_str(self, df_source1, df_source2):
        df_source = df_source1.reindex(index=df_source2.index, columns=df_source2.columns)
        result = copy(df_source)
        for di in df_source.index:
            for dj in df_source.columns:
                move = df_source2.loc[di, dj].astype('int')
                day = self.before(di, num=move)
                if day in df_source.index:
                    df = df_source.loc[day:di, dj].rank()
                    result.loc[di, dj] = df[di]
        return result


def sxy_equalzero_str(df_source, return1, return2):
    mtx = pd.DataFrame(index=df_source.index, columns=df_source.columns, data=np.nan)
    mtx[df_source == 0] = return1[df_source == 0]
    mtx[df_source != 0] = return2[df_source != 0]
    return mtx


def self_std_str(df_source, last, now):
    mtx = df_source.loc[last:now]
    return mtx.std(axis=0)


def self_mean_str(df_source, last, now):
    mtx = df_source.loc[last:now]
    return mtx.mean(axis=0)


if __name__ == '__main__':
    # the current version of data_update do not support the function of info updating daily,
    # the second version may be improved

    sxy_alpha = Alpha(config)
    sxy_alpha.load_data_from_local()

    sxy_alpha.wavelet_v1()
    sxy_alpha.wavelet_v2()
    sxy_alpha.wavelet_v3()
    wavelet_name = 'wavelet_v3'
    print('data ready')
    sxy_alpha.sxy_temp0(wavelet_name)
    sxy_alpha.sxy_temp1(wavelet_name)
    sxy_alpha.sxy_temp2(wavelet_name)
    sxy_alpha.sxy_temp3(wavelet_name)
    sxy_alpha.sxy_temp4(wavelet_name)











