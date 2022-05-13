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


# reload(sys)
# sys.setdefaultencoding('utf-8')


class Alpha(Alpha_Base):

    def __init__(self, cf):
        Alpha_Base.__init__(self, cf)
        self.h5_path_5min = cf['h5_path_5min']
        self.h5_path_1min = cf['h5_path_1min']
        self.h5_path_tick = cf['h5_path_tick']

    #        data_path     = root_path+r'/workspace_data'
    def load_other_data(self):
        print('load data from other resources')
        root_path = r'/home/xysong/PRODUCT'
        data_path = root_path + r'/workspace_data'
        #        self.oper_net_cf=QLLIB.read_from_selflib(data_name = 'oper_net_cash_flow',\
        #                data_path = data_path).reindex(columns = self.columns)
        #        self.oper_rev_ttm_yoy=QLLIB.read_from_selflib(data_name = 'oper_rev_ttm_yoy',\
        #                data_path = data_path).reindex(columns = self.columns)
        #        self.e_ttm=QLLIB.read_from_selflib(data_name = 'e_ttm',\
        #                data_path = data_path).reindex(columns = self.columns)
        self.e_fa = QLLIB.read_from_selflib(data_name='e_fa', \
                                            data_path=data_path).reindex(columns=self.columns)
        #        self.e_qfa_yoy=QLLIB.read_from_selflib(data_name = 'e_qfa_yoy_dedu',\
        #                data_path = data_path).reindex(columns = self.columns)
        self.tot_asset = QLLIB.read_from_selflib(data_name='tot_assets', \
                                                 data_path=data_path).reindex(columns=self.columns)
        #        self.tot_liability=QLLIB.read_from_selflib(data_name = 'tot_liab',\
        #                data_path = data_path).reindex(columns = self.columns)
        #        self.rev_ttm=QLLIB.read_from_selflib(data_name = 'oper_rev_ttm',\
        #                data_path = data_path).reindex(columns = self.columns)

        #        self.SMALL_BUY=QLLIB.read_from_selflib(data_name = 'BUY_VOLUME_SMALL_ORDER',\
        #                data_path = data_path).reindex(columns = self.columns)
        self.INST_BUY = (QLLIB.read_from_selflib(data_name='BUY_VOLUME_EXLARGE_ORDER', \
                                                 data_path=data_path) + QLLIB.read_from_selflib(
            data_name='BUY_VOLUME_LARGE_ORDER', \
            data_path=data_path)).reindex(columns=self.columns)

    #        self.SMALL_DIFF=QLLIB.read_from_selflib(data_name = 'VOLUME_DIFF_SMALL_TRADER',\
    #                data_path = data_path).reindex(columns = self.columns)
    #        self.MED_DIFF=QLLIB.read_from_selflib(data_name = 'VOLUME_DIFF_MED_TRADER',\
    #                data_path = data_path).reindex(columns = self.columns)
    #        self.INST_DIFF=QLLIB.read_from_selflib(data_name = 'VOLUME_DIFF_INSTITUTE',\
    #                data_path = data_path).reindex(columns = self.columns)

    #        style_ts.to_csv(data_path_group+r'/PE_chgRate_style_ts.csv')
    #        for i_dt in map_dict.keys():
    #            map_dict[i_dt].to_csv(data_path_group+r'/PE_grouping/%s.csv'%i_dt)

    def cal_grouping_data(self):
        self.map_dict, self.style_ts = QLLIB.generate_ts_map_info(self.cap_p, \
                                                                  self.chgRate_p, define_central='median', level_num=20)
        root_path = r'/home/xysong/PRODUCT'
        data_path_group = root_path + r'/workspace_intern/Group_data'

        self.style_ts.to_csv(data_path_group + r'/cap_chgRate_style_ts.csv')
        for i_dt in self.map_dict.keys():
            self.map_dict[i_dt].to_csv(data_path_group + r'/cap_grouping/%s.csv' % i_dt)

    def load_grouping_data(self, method):

        root_path = r'/home/xysong/PRODUCT'
        data_path_group = root_path + r'/workspace_intern/Group_data'
        #        self.style_ts=QLLIB.read_from_selflib(data_name = method+'_chgRate_style_ts',\
        #                                         data_path = data_path_group)
        self.map_dict = {}
        for i_dt in self.index:
            self.map_dict[i_dt] = QLLIB.read_from_selflib(data_name='%s' % i_dt, \
                                                          data_path=data_path_group + r'/' + method + '_grouping')

    # 计算因子风格分组
    def factor_grouping(self):
        root_path = r'/home/xysong/PRODUCT'
        data_path = root_path + r'/workspace_data'
        data_path_group = root_path + r'/workspace_intern/Group_data'
        e_qfa_yoy = QLLIB.read_from_selflib(data_name='e_qfa_yoy_dedu', data_path=data_path).reindex(
            columns=self.columns)
        e_ttm = QLLIB.read_from_selflib(data_name='e_ttm', data_path=data_path).reindex(columns=self.columns)
        BETA = QLLIB.read_from_selflib(data_name='Fbeta_self', data_path=data_path).reindex(columns=self.columns)
        oper_net_cf = QLLIB.read_from_selflib(data_name='oper_net_cash_flow', data_path=data_path).reindex(
            columns=self.columns)
        tot_asset = QLLIB.read_from_selflib(data_name='tot_assets', data_path=data_path).reindex(columns=self.columns)
        ST_filter = QLLIB.read_from_selflib(data_name='filter_ST', data_path=data_path).reindex(columns=self.columns)

        self.map_dict = {}
        type_index = ['type_%d' % i for i in range(24)]
        factor_dict = {}

        beta_rank = self_rank_mtx(BETA)
        factor_dict['type_0'] = beta_rank
        factor_dict['type_1'] = 1 - beta_rank
        #        factor_rank.append(beta_rank)
        #        factor_rank.append(1-beta_rank)

        PE_rank = self_rank_mtx(self.PE_p)
        factor_dict['type_2'] = PE_rank
        factor_dict['type_3'] = 1 - PE_rank
        #        factor_rank.append(PE_rank)
        #        factor_rank.append(1-PE_rank)

        SIZE_rank = self_rank_mtx(self.SIZE_EXP)
        factor_dict['type_4'] = SIZE_rank
        factor_dict['type_5'] = 1 - SIZE_rank
        #        factor_rank.append(SIZE_rank)
        #        factor_rank.append(1-SIZE_rank)

        NLSIZE_rank = self_rank_mtx(self.NLSIZE_EXP)
        factor_dict['type_6'] = NLSIZE_rank
        factor_dict['type_7'] = 1 - NLSIZE_rank
        #        factor_rank.append(NLSIZE_rank)
        #        factor_rank.append(1-NLSIZE_rank)

        LEVERAGE_rank = self_rank_mtx(self.LEVERAGE_EXP)
        factor_dict['type_8'] = LEVERAGE_rank
        factor_dict['type_9'] = 1 - LEVERAGE_rank
        #        factor_rank.append(LEVERAGE_rank)
        #        factor_rank.append(1-LEVERAGE_rank)

        VOLATILITY_rank = self_rank_mtx(self.VOLATILITY_EXP)
        factor_dict['type_10'] = VOLATILITY_rank
        factor_dict['type_11'] = 1 - VOLATILITY_rank
        #        factor_rank.append(VOLATILITY_rank)
        #        factor_rank.append(1-VOLATILITY_rank)

        GROWTH_rank = self_rank_mtx(self.GROWTH_EXP)
        factor_dict['type_12'] = GROWTH_rank
        factor_dict['type_13'] = 1 - GROWTH_rank
        #        factor_rank.append(GROWTH_rank)
        #        factor_rank.append(1-GROWTH_rank)

        QUALITY_rank = self_rank_mtx(oper_net_cf / tot_asset)
        factor_dict['type_14'] = QUALITY_rank
        factor_dict['type_15'] = 1 - QUALITY_rank
        #        factor_rank.append(QUALITY_rank)
        #        factor_rank.append(1-QUALITY_rank)

        LIQUIDITY_rank = self_rank_mtx(self.LIQUIDITY_EXP)
        factor_dict['type_16'] = LIQUIDITY_rank
        factor_dict['type_17'] = 1 - LIQUIDITY_rank
        #        factor_rank.append(LIQUIDITY_rank)
        #        factor_rank.append(1-LIQUIDITY_rank)

        EARNING_rank = self_rank_mtx(e_qfa_yoy)
        factor_dict['type_18'] = EARNING_rank
        #        factor_rank.append(EARNING_rank)

        SHORTMOM_rank = self_rank_mtx(self.SHORTTERMMOMENTUM_EXP)
        factor_dict['type_19'] = SHORTMOM_rank
        factor_dict['type_20'] = 1 - SHORTMOM_rank
        #        factor_rank.append(SHORTMOM_rank)
        #        factor_rank.append(1-SHORTMOM_rank)

        MEDMOM_rank = self_rank_mtx(self.MEDIUMTERMMOMENTUM_EXP)
        factor_dict['type_21'] = MEDMOM_rank
        factor_dict['type_22'] = 1 - MEDMOM_rank
        #        factor_rank.append(MEDMOM_rank)
        #        factor_rank.append(1-MEDMOM_rank)

        ST_f = ST_filter.isna()
        PE_f = (self.PE_p > 600) | (self.PE_p < 0)
        E_f = e_ttm / tot_asset < -0.2
        TRASH = ST_f | PE_f | E_f
        factor_dict['type_23'] = TRASH.astype("int") * 2

        for di in self.index[:]:
            print(di)
            grouping = pd.DataFrame(index=type_index, columns=self.columns, data=np.nan)
            for i_key in type_index:
                grouping.loc[i_key] = factor_dict[i_key].loc[di]
            grouping = grouping.rank(axis=0)

            grouping = grouping[grouping > 23]
            #            grouping = grouping[grouping=grouping.max(axis=0)]
            grouping = grouping * 0 + 1

            self.map_dict[di] = grouping

        def mkdir(path):
            folder = os.path.exists(path)
            if not folder:
                os.makedirs(path)

        mkdir(data_path_group + r'/factor_grouping')
        for i_dt in self.map_dict.keys():
            self.map_dict[i_dt].to_csv(data_path_group + r'/factor_grouping/%s.csv' % i_dt)

    def demo_004(self):
        #        root_path=r'/home/xysong/PRODUCT'
        #        data_path_group = root_path+r'/workspace_intern/Group_data'
        #        self.style_ts=QLLIB.read_from_selflib(data_name = method+'_chgRate_style_ts',\
        #                                         data_path = data_path_group)
        config = {}
        config['alpha_name'] = 'demo_004'
        config['alpha_num'] = 'demo_004'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_5 = self.trade_day[axis_now - 20 + 1]

            signal_vect = self_sum_str(self.chgRate_p, axis_5, di) * univers
            tot_num_each_group = self.map_dict[di].sum(axis=1)
            first_500_stocks = signal_vect > signal_vect.quantile(0.87)
            temp = [0 for i in range(len(self.map_dict[di]))]
            for ticker in self.columns:
                if first_500_stocks[ticker]:
                    for i in range(len(self.map_dict[di])):
                        if self.map_dict[di][ticker][i] == 1:
                            temp[i] += 1
                            break
            f5s_num_each_group = pd.Series(temp, index=self.map_dict[di].index)
            style_signal = f5s_num_each_group / tot_num_each_group

            #            for ticker in self.columns:

            #            vect_alpha  = QLLIB.map_from_mtx(map_df = self.ind_dict[di],\
            #                factor_vect = signal_vect,retn_type = 'map',\
            #                define_central = 'median')*univers
            # vect_alpha  = std_vect(vect_alpha)
            vect_alpha = QLLIB.map_from_style(self.map_dict[di], style_signal) * univers
            # vect_alpha  = std_vect(vect_alpha)
            m = result.loc[di]
            m[:] = vect_alpha
        return result, config

    #####################  ###################
    def demo_001(self):
        # 简单操作，依据两个信号直接合成板块信号，可用于计算板块动量等简单信号
        # 主要用于依据特征排序进行分类的场景
        # 主要函数: map_from_signal()
        # 市值作为分类风格、5日涨幅作为信号、分20档、板块内信号中位数作为板块信号、信号生成方式为map
        # retn_type : map 表示把板块信号映射到板块上每个股票；delta 表示个股信号与板块信号的差
        # define_central: median 表示取中位数作为板块信号， mean 表示取均值作为板块信号
        # level_num : 根据style_vect 分类的数量
        config = {}
        config['alpha_name'] = 'demo_001'
        config['alpha_num'] = 'demo_001'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        close_re = (self.close_p * self.re_p).rolling(window=20, min_periods=10).mean()

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_5 = self.trade_day[axis_now - 5 + 1]

            style_vect = self.cap_p.loc[di] * univers
            signal_vect = self_sum_str(self.chgRate_p, axis_5, di) * univers

            vect_alpha = QLLIB.map_from_signal(style_vect=style_vect, factor_vect=signal_vect, \
                                               retn_type='map', define_central='median', level_num=20) * univers
            # vect_alpha  = std_vect(vect_alpha)
            m = result.loc[di]
            m[:] = vect_alpha
        return result, config

    def demo_002(self):
        # 简单操作，依据分类信号将类别信息映射到每只股票，可用于计算板块动量等简单信号
        # 主要用于一些自定义的分类、行业分类等无法用特征排序的场景
        # 需要提前做好分类信息，输入的类别信息必须是哑变量矩阵，比如行业分类(分类信息一般每天一矩阵)
        # 主要函数: map_from_mtx()
        # 市值作为分类风格、5日涨幅作为信号、分20档、板块内信号中位数作为板块信号、信号生成方式为map
        # retn_type : map 表示把板块信号映射到板块上每个股票；delta 表示个股信号与板块信号的差
        # define_central: median 表示取中位数作为板块信号， mean 表示取均值作为板块信号
        config = {}
        config['alpha_name'] = 'demo_002'
        config['alpha_num'] = 'demo_002'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_5 = self.trade_day[axis_now - 5 + 1]

            signal_vect = self_sum_str(self.chgRate_p, axis_5, di) * univers

            vect_alpha = QLLIB.map_from_mtx(map_df=self.ind_dict[di], \
                                            factor_vect=signal_vect, retn_type='map', \
                                            define_central='median') * univers
            # vect_alpha  = std_vect(vect_alpha)
            m = result.loc[di]
            m[:] = vect_alpha
        return result, config

    def demo_003(self):
        # 复杂操作，根据分类风格和股票信号，先生成按天存储的股票分类映射字典（类似行业分类，每天有一个分类矩阵）
        # 和每个分类信号的时间序列，再基于该时间序列做复杂计算，比如计算均线、相关性、换手变化等
        # 最后，基于映射字典将计算出的复杂信号映射到每只股票上
        # 主要函数：generate_ts_map_info()  map_from_style()
        # 市值作为分类风格、计算每个类别涨幅的20日自相关性：先得到每个类别涨幅的时间序列->
        # 计算每个序列自相关性->映射回股票

        # retn_type : map 表示把板块信号映射到板块上每个股票；delta 表示个股信号与板块信号的差
        # define_central: median 表示取中位数作为板块信号， mean 表示取均值作为板块信号
        # level_num : 根据style_vect 分类的数量
        config = {}
        config['alpha_name'] = 'demo_003'
        config['alpha_num'] = 'demo_003'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        # 此处self.cap_p,self.chgRate_p都是系统自带数据，如果要用自己计算的风格因子和信号，
        # 需要先算出相应信号，再放到下面函数中。输入的两个dataframe的index 日期必须一致。
        map_dict, style_ts = QLLIB.generate_ts_map_info(self.cap_p, \
                                                        self.chgRate_p, define_central='median', level_num=20)

        # 研究中可将中间结果保存到本地,重复利用，不必每次都重新生成。
        # 读文件的代码：result = QLLIB.read_from_selflib(data_name = 'close_p',data_path = '')
        # 上述代码要求被读入的文件必须有index和columns标签，如果没有，就自己写读文件的代码
        # style_ts.to_csv(……) ，
        # for i_dt in map_dict.keys():
        #    map_dict[i_dt].to_csv('……/%s.csv'%i_dt)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_20 = self.trade_day[axis_now - 20 + 1]

            map_df = map_dict[di]
            style_signal = QLLIB.self_corr_str(style_ts, style_ts.shift(1), axis_20, di)

            vect_alpha = QLLIB.map_from_style(map_df, style_signal) * univers
            # vect_alpha  = std_vect(vect_alpha)
            m = result.loc[di]
            m[:] = vect_alpha
        return result, config

    def alpha_ype_001(self):
        config = {}
        config['alpha_name'] = 'alpha_ype_001'
        config['alpha_num'] = 'alpha_ype_001'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_15 = self.trade_day[axis_now - 15 + 1]

            style_vect = self.turnover_p.loc[di] * univers
            signal_vect = -self_skew_str(self.chgRate_p, axis_15, di) * univers

            vect_alpha = QLLIB.map_from_signal(style_vect=style_vect, factor_vect=signal_vect, \
                                               retn_type='map', define_central='median', level_num=20) * univers
            # vect_alpha  = std_vect(vect_alpha)
            m = result.loc[di]
            m[:] = vect_alpha
        return result, config

    def alpha_ype_002(self):
        config = {}
        config['alpha_name'] = 'alpha_ype_002'
        config['alpha_num'] = 'alpha_ype_002'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_15 = self.trade_day[axis_now - 15 + 1]

            style_vect = self.SHORTTERMMOMENTUM_EXP.loc[di] * univers
            signal_vect = -self_skew_str(self.chgRate_p, axis_15, di) * univers

            vect_alpha = QLLIB.map_from_signal(style_vect=style_vect, factor_vect=signal_vect, \
                                               retn_type='map', define_central='median', level_num=20) * univers
            # vect_alpha  = std_vect(vect_alpha)
            m = result.loc[di]
            m[:] = vect_alpha
        return result, config

    def alpha_ype_003(self):
        config = {}
        config['alpha_name'] = 'alpha_ype_003'
        config['alpha_num'] = 'alpha_ype_003'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        for di in self.index:
            univers = self.univers.loc[di]
            signal_vect = -self.INST_BUY.loc[di] / self.volume_p.loc[di] * univers
            vect_alpha = QLLIB.map_from_mtx(map_df=self.map_dict[di], \
                                            factor_vect=signal_vect, retn_type='map', \
                                            define_central='median') * univers
            m = result.loc[di]
            m[:] = vect_alpha
        return result, config

    def alpha_ype_004(self):
        config = {}
        config['alpha_name'] = 'alpha_ype_004'
        config['alpha_num'] = 'alpha_ype_004'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_120 = self.trade_day[axis_now - 120 + 1]

            signal_vect = -self_tsrank_str(self.PE_p, axis_120, di) * univers
            vect_alpha = QLLIB.map_from_mtx(map_df=self.map_dict[di], \
                                            factor_vect=signal_vect, retn_type='map', \
                                            define_central='median') * univers
            m = result.loc[di]
            m[:] = vect_alpha
        return result, config

    def alpha_ype_005(self):
        config = {}
        config['alpha_name'] = 'alpha_ype_005'
        config['alpha_num'] = 'alpha_ype_005'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_60 = self.trade_day[axis_now - 60 + 1]
            price_chgRate_q = self_sum_str(self.chgRate_p, axis_60, di)
            earn_q = self.e_fa.loc[di] / self.tot_asset.loc[di]
            #            signal_vect = (self_sum_str(self.chgRate_p,axis_10,di)-self_sum_str(self.chgRate_p,axis_5,di))*univers
            signal_vect = (earn_q - price_chgRate_q) * univers
            vect_alpha = QLLIB.map_from_mtx(map_df=self.map_dict[di], \
                                            factor_vect=signal_vect, retn_type='map', \
                                            define_central='median') * univers

            m = result.loc[di]
            m[:] = vect_alpha

        return result, config

    def alpha_ype_006(self):
        config = {}
        config['alpha_name'] = 'alpha_ype_006'
        config['alpha_num'] = 'alpha_ype_006'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        for di in self.index:
            univers = self.univers.loc[di]
            signal_vect = self.chgRate_p.loc[di] * univers
            vect_alpha = QLLIB.map_from_mtx(map_df=self.map_dict[di], \
                                            factor_vect=signal_vect, retn_type='map', \
                                            define_central='median') * univers

            m = result.loc[di]
            #            m[:] = QLLIB.std_vect(vect_alpha)
            #            m[:] = signal_vect
            m[:] = vect_alpha
        return result, config

    def alpha_ype_007(self):
        config = {}
        config['alpha_name'] = 'alpha_ype_007'
        config['alpha_num'] = 'alpha_ype_007'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_20 = self.trade_day[axis_now - 20 + 1]

            signal_vect = self_sum_str(self.chgRate_p, axis_20, di) * univers

            vect_alpha = QLLIB.map_from_mtx(map_df=self.map_dict[di], \
                                            factor_vect=signal_vect, retn_type='map', \
                                            define_central='median') * univers

            m = result.loc[di]
            #            m[:] = QLLIB.std_vect(vect_alpha)
            #            m[:] = signal_vect
            m[:] = -vect_alpha
        return result, config

    def alpha_ype_008(self):
        config = {}
        config['alpha_name'] = 'alpha_ype_008'
        config['alpha_num'] = 'alpha_ype_008'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        #        close_re = (self.close_p*self.re_p)

        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_5 = self.trade_day[axis_now - 3 + 1]
            dict_temp = self.map_dict[di].fillna(0)

            signal_vect = self_sum_str(self.chgRate_p, axis_5, di) * univers
            tot_num_each_group = dict_temp.sum(axis=1)
            first_500_stocks = (signal_vect > signal_vect.quantile(0.87)).astype("int")

            f5s_num_each_group = dict_temp.dot(first_500_stocks)
            style_signal = -f5s_num_each_group / tot_num_each_group

            vect_alpha = QLLIB.map_from_style(dict_temp, style_signal) * univers
            m = result.loc[di]
            m[:] = vect_alpha
        return result, config

    def alpha_ype_009(self):
        config = {}
        config['alpha_name'] = 'alpha_ype_009'
        config['alpha_num'] = 'alpha_ype_009'
        config['decay_days'] = 5
        config['res_flag'] = 0
        result = pd.DataFrame(index=self.index, columns=self.columns, data=np.nan)
        close_re = (self.close_p * self.re_p)
        #        close_tsrank=self_rank_mtx(close_re.T).T
        for di in self.index:
            univers = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_260 = self.trade_day[axis_now - 120 + 1]
            axis_20 = self.trade_day[axis_now - 20 + 1]

            one_year_high = close_re.loc[axis_260:di].max(axis=0)
            one_month_high = close_re.loc[axis_20:di].max(axis=0)
            #            one_year_low = close_re.loc[axis_260:di].min(axis=0)
            signal_vect = one_month_high / one_year_high
            adj_ind_dict = self.ind_dict[di].reindex(columns=self.columns).fillna(0)
            tot_num_each_group = adj_ind_dict.sum(axis=1)

            first_500_stocks = (signal_vect > 0.90).astype("int")
            f5s_num_each_group = adj_ind_dict.dot(first_500_stocks)
            style_signal = f5s_num_each_group / tot_num_each_group

            vect_alpha = -QLLIB.map_from_style(adj_ind_dict, style_signal) * univers
            m = result.loc[di]
            m[:] = vect_alpha.reindex(self.columns)
        #            m[:] = vect_alpha
        return result, config


def ind_rank(df_source, ind_ref):
    # 行业内rank
    ind_source = ind_ref * df_source
    rank_result = ind_source.rank(axis=1)
    max_value = rank_result.max(axis=1)
    result = ((rank_result.T) / max_value).T
    return result.sum()


def self_mean_str(df_source, last, now):
    mtx = df_source.loc[last:now]
    return mtx.mean(axis=0)


def self_sum_str(df_source, last, now):
    mtx = df_source.loc[last:now]
    return mtx.sum(axis=0)


def self_kurt_str(df_source, last, now):
    mtx = df_source.loc[last:now]
    return mtx.kurtosis(axis=0)


def self_skew_str(df_source, last, now):
    mtx = df_source.loc[last:now]
    return mtx.skew(axis=0)


def self_tsrank_str(df_source, last, now):
    mtx = df_source.loc[last:now]
    return mtx.rank().loc[now]


def self_max_str(df_source, last, now):
    mtx = df_source.loc[last:now]
    return mtx.max(axis=0)


def self_idxmax_str_np(df_source, last, now):
    mtx = df_source.loc[last:now]
    vect = np.argsort(-mtx.values, axis=0)[0, :]
    return pd.Series(vect, index=df_source.columns)


def self_idxmax_str(df_source, last, now):
    mtx = pd.DataFrame(df_source.loc[last:now].values, columns=df_source.columns)
    return mtx.idxmax(axis=0)


def self_idxmin_str(df_source, last, now):
    mtx = pd.DataFrame(df_source.loc[last:now].values, columns=df_source.columns)
    return mtx.idxmin(axis=0)


def z_score_org(vect):
    return (vect - vect.mean()) / vect.std()


def self_min_str(df_source, last, now):
    mtx = df_source.loc[last:now]
    return mtx.min(axis=0)


def self_std_str(df_source, last, now):
    mtx = df_source.loc[last:now]
    return mtx.std(axis=0)


def self_sigmoid(vect):
    result = 1 / (1 + np.exp(-vect))
    return result


def MAD_Outlier(arr):
    arr = arr.astype(float)
    if sum(np.isnan(arr.astype(float))) == len(arr):
        return arr
    median = np.nanmedian(arr)
    MAD = np.nanmedian(np.abs(arr - median))
    arr[arr > median + 6 * 1.4826 * MAD] = median + 6 * 1.4826 * MAD
    arr[arr < median - 6 * 1.4826 * MAD] = median - 6 * 1.4826 * MAD
    return arr


def self_decay_equal(df_source, di, length=10):
    if di < length - 1:
        length = di + 1

    new_source = copy(df_source.iloc[di - length + 1:di + 1])

    new_source = new_source.mean()
    return new_source


def std_vect_org(vect):
    result = (vect - vect.mean()) / vect.std()
    return result


def std_vect(vect, level=10):
    med = vect.median()
    err = (vect - med).abs().median()
    up_limite = med + level * err
    down_limite = med - level * err
    vect[vect > up_limite] = up_limite
    vect[vect < down_limite] = down_limite
    result = (vect - vect.mean()) / vect.std()
    return result


def std_vect_mtx(df_source, level=6):
    result = pd.DataFrame(index=df_source.index, columns=df_source.columns, data=np.nan)
    for di in df_source.index:
        vect = df_source.loc[di]
        vect = std_vect(vect)
        m = result.loc[di]
        m[:] = vect
    return result


def std_vect_mean(vect, level=20):
    med = vect.mean()
    err = (vect - med).abs().std()
    up_limite = med + level * err
    down_limite = med - level * err
    vect[vect > up_limite] = up_limite
    vect[vect < down_limite] = down_limite
    result = (vect - vect.mean()) / vect.std()
    return result


def self_rank_mtx(df_source):
    df_rank = df_source.rank(axis=1)
    max_value = df_rank.max(axis=1)
    df_rank = ((df_rank.T) / max_value).T
    return df_rank


def self_WMA(df_source, last, now, num=1):
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


def self_EMA(df_source, last, now):
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


def self_decay_v1(df_source, di, length=10):
    if di < length - 1:
        length = di + 1
    ratio = pd.Series(range(length + 1)[1:])
    new_source = (copy(df_source.iloc[di - length + 1:di + 1]).T)
    m = pd.Series(index=new_source.columns)
    m[:] = ratio
    new_source = new_source * m
    new_source = new_source.sum(axis=1) / (length * (length + 1) * 0.5)
    return new_source.T


def self_decay_old(df_source, di, length=10):
    if di < length - 1:
        length = di + 1
    new_source = copy(df_source.iloc[di - length + 1:di + 1])
    for i in range(length):
        new_source.iloc[i] = new_source.iloc[i] * (i + 1)
    return new_source.sum(axis=0) / (length * (length + 1) * 0.5)


def self_std(df_source, di, length=20):
    if di < length - 1:
        length = di + 1
    new_source = df_source.iloc[di - length + 1:di + 1]
    return new_source.std(axis=0)  # 求列方差


def self_corr(df_source_1, df_source_2, di, length=20):
    if di < length - 1:
        print('warning the di < data_length')
        length = di + 1
    new_source_1 = df_source_1.iloc[di - length + 1:di + 1]
    new_source_2 = df_source_2.iloc[di - length + 1:di + 1]
    mult = new_source_1 * new_source_2
    cov = mult.mean(axis=0) - new_source_1.mean(axis=0) * new_source_2.mean(axis=0)
    std = new_source_1.std(axis=0) * new_source_2.std(axis=0) + 0.0001
    return cov / std


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


def self_rank(df_source):
    df_source = df_source.rank()
    min_value = df_source.min()
    max_value = df_source.max()
    if abs(max_value - min_value).any() < 0.001:
        print('rank error, the max_value = min_value')
        return df_source * 0
    return (df_source - min_value) / (max_value - min_value)


def self_normalize(df_source):
    min_value = df_source.min(axis=1)
    max_value = df_source.max(axis=1)
    if abs(max_value - min_value).any() < 0.0000001:
        print('rank error, the max_value = min_value')
        return df_source * 0
    return (df_source - min_value) / (max_value - min_value)


def rank(df_source):
    # 与self_rank
    df_source = df_source.rank()
    min_value = df_source.min()
    max_value = df_source.max()
    if abs(max_value - min_value) < 0.001:
        print('rank error, the max_value = min_value')
        return df_source * 0
    return (df_source - min_value) / (max_value - min_value)

    #


def ind_neutral(df_source, ind_p):
    result = ind_p * df_source

    mean = result.mean()
    std = result.std() + 0.000001
    result = (result.T - mean) / std

    result = (result.T).sum(axis=0)
    result[result == 0] = np.nan
    return result


def self_free_neutral(df_source, objt):
    result = objt * df_source
    mean = result.mean()
    std = result.std() + 0.000001
    result = (result.T - mean) / std

    result = (result.T).sum(axis=0)
    result[result == 0] = np.nan
    return result


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













































