"""
Created on Thu Mar 21 17:05:26 2019

@author: YoYo
"""

#---------------------- 本文件用于跑risk factor单因子回归，并输出因子有效性检验的参数表格，表格格式与报告完全一致 ----------------------#


import pandas as pd
import validity
import os
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
from pyecharts import Line
from base_class_modified import base_class
import logging
import warnings
import logging.config
from awesome_functions import fill_by_category

os.chdir("E:\\python\\project\\My_Barra")

warnings.filterwarnings("ignore")
logging.config.fileConfig("log.conf")
log = logging.getLogger("info")

"""#####################读取基本变量#####################"""
start_date_time = '20100129'
end_date_time = '20150129'

Return = pd.read_csv("./factor/Return.csv",index_col=0)
Return = (Return.shift(periods=-2,axis=1)+1) * (Return.shift(periods=-2,axis=1)+1) - 1     # t+2对t回归
time_list = list(Return.loc[:,start_date_time:end_date_time].columns)

start_point = list(Return.columns).index(start_date_time)
end_point = list(Return.columns).index(end_date_time)

"""求月度收益率"""
# 阅读收益率按照下面的算法求出，并且保存，因此只需读取即可
# Return = (Return + 1).cumprod(1)
# # mon_return = pd.DataFrame(index = Return.index,columns = Return.columns)
# for i in range(21,len(Return.columns)):
#     mon_return.iloc[:,i - 21] = Return.iloc[:,i] / Return.iloc[:,i - 21] - 1
# Return = mon_return
Return_month = pd.read_csv("./factor/Return_month.csv",index_col = 0)

industry_name_list = ['交通运输', '休闲服务', '传媒', '公用事业', '农林牧渔', '化工', '医药生物', '商业贸易',
                      '国防军工', '家用电器', '建筑材料', '建筑装饰', '房地产开发Ⅱ', '园区开发Ⅱ',
                      '有色金属', '机械设备', '汽车', '电子', '电气设备', '纺织服装', '综合', '计算机',
                      '轻工制造', '通信', '采掘', '钢铁', '银行', '保险Ⅱ', '证券Ⅱ', '多元金融Ⅱ',
                      '食品加工', '饮料制造']
risk_factor_list = ['beta','momentum','size','earning_yield','value','volatility','liquidity','leverage','growth']

base_instance = base_class(start_point,end_point,risk_factor_list)
industry = base_instance.other_factors['industry'].reindex(index=Return.index)
weight = base_instance.other_factors['weight'].reindex(index=Return.index)

"""建立待求变量空壳"""

factor_list = risk_factor_list + industry_name_list
t_df = pd.DataFrame(index=factor_list, columns=time_list)
coef_df = pd.DataFrame(index=factor_list, columns=time_list)
std_error_df = pd.DataFrame(index=factor_list, columns=time_list)
factor_stability_coef = pd.DataFrame(index=["factor_stability_coef"],columns=risk_factor_list)

"""#####################单因子回归#####################"""

def singles_risk_tet(Return,file_name,period):
    for i in tqdm(risk_factor_list):
        index = risk_factor_list.index(i)
        current_risk_factor = base_instance.risk_factors[index]
        current_risk_factor = current_risk_factor.loc[:,start_date_time:end_date_time]              # 只取和报告一致的时间段

        for j in current_risk_factor.columns:
            if current_risk_factor[j].dropna().empty == False:  # 判断当天的因子是否有值

                current_return = Return[j].dropna(axis = 0)
                current_column = current_risk_factor[j].reindex(index=current_return.index)
                current_column = current_column.fillna(current_column.mean())
                current_column = sm.add_constant(current_column)
                current_weight = weight[j].fillna(weight[j].mean())

                data = pd.concat([current_return, current_column, current_weight], join='inner', axis=1, sort=True) # 整合同一天的return（t+2）、risk factor和weight
                # data = data.dropna(how = "any",axis = 0)  # 去掉空值才能回归

                log.debug(f"The number observations for {i}-{j} is {data.shape[0]}")

                result = base_class.regression(data.iloc[:,1:3],data.iloc[:,0],data.iloc[:,3])  # 调用base class里的regression函数做 wls 回归
                t_df.loc[i,j] = result.tvalues[1]                # 取回归当天的t值
                coef_df.loc[i,j] = result.params[1]/period       # 取回归当天的coef
                std_error_df.loc[i,j] = result.bse[1]/period     # 取回归当天的标准差

    for i in tqdm(range(len(industry.columns))):
        current_industry = industry.iloc[:,i]                                       # 当前行业，对每天回归时保持不变
        current_return = Return.loc[:,start_date_time:end_date_time]

        for j in range(len(current_return.columns)):
            data_y = current_return.iloc[:,j].dropna()
            weight_temp = weight.iloc[:,j].reindex(index=data_y.index)
            weight_temp = weight_temp.fillna(weight_temp.mean())
            data_x = current_industry.reindex(data_y.index)
            data_x = data_x.fillna(data_x.mean())
            data_x = sm.add_constant(data_x)

            data = pd.concat([data_y,data_x,weight_temp],join='inner', axis=1)

            # data = data.dropna(how="any", axis=0)  # 去掉空值才能回归

            logging.debug(f"The number observations for {industry_name_list[i]}-{time_list[j]} is {data.shape[0]}")
            result = base_class.regression(data.iloc[:, 1:3], data.iloc[:, 0],
                                           data.iloc[:, 3])  # 调用base class里的regression函数做 wls 回归
            industry_index = len(risk_factor_list) + i
            t_df.iloc[industry_index, j] = result.tvalues[1]              # 取回归当天的t值
            coef_df.iloc[industry_index, j] = result.params[1]/period     # 取回归当天的coef
            std_error_df.iloc[industry_index, j] = result.bse[1]/period   # 取回归当天的标准差

    t_df.to_csv("./output/t_df_" + file_name + ".csv",encoding="gbk")
    coef_df.to_csv("./output/coef_df_"+ file_name + ".csv",encoding="gbk")

"""#####################单因子+行业回归#####################"""

def single_risk_industry_test(Return,file_name,period):

    for i in tqdm(risk_factor_list):
        index = risk_factor_list.index(i)
        current_risk_factor = base_instance.risk_factors[index]
        current_risk_factor = current_risk_factor.loc[:,start_date_time:end_date_time]              # 只取和报告一致的时间段

        for j in current_risk_factor.columns:
            if current_risk_factor[j].dropna().empty == False:  # 判断当天的因子是否有值

                current_return = Return[j].dropna(axis = 0)
                current_column = current_risk_factor[j].reindex(index=current_return.index)
                current_weight = weight[j]
                data = pd.concat([current_return, current_column,industry,current_weight], join='inner', axis=1, sort=True) # 整合同一天的return（t+2）、risk factor和weight
                data = fill_by_category(data, [1,-1], [2, -1])

                log.debug(f"The number observations for {i}-{j} is {data.shape[0]}")

                result = base_class.regression(data.iloc[:,1:-1],data.iloc[:,0],data.iloc[:,-1])  # 调用base class里的regression函数做 wls 回归
                t_df.loc[i,j] = result.tvalues[0]                # 取回归当天的t值
                coef_df.loc[i,j] = result.params[0]/period       # 取回归当天的coef
                std_error_df.loc[i,j] = result.bse[0]/period     # 取回归当天的标准差

    t_df.to_csv("./output/t_df_" + file_name + ".csv",encoding="gbk")
    coef_df.to_csv("./output/coef_df_"+ file_name + ".csv",encoding="gbk")

"""#####################多因子回归#####################"""

def multi_risk_test(Return,file_name,period):
    """
    多因子回归，获得t值、因子收益率等
    :param Return: 多元回归使用的收益率
    :param file_name: 每次回归的名字，用于生成文件的命名
    :param period: 收益率的时间长度
    """

    factor_scs = []

    for date in tqdm(time_list):

        risk_columns = []
        current_return = Return[date]
        current_return = current_return.dropna(how = "any",axis = 0)     # 删除当天return为空的样本

        for x in base_instance.risk_factors:
            current_x = x[date].reindex(index = current_return.index)    # 删除当天return为空的样本
            risk_columns.append(current_x)                               # 提取risk的数据

        risk_columns = pd.concat(risk_columns, axis=1, sort=True)
        current_industry = industry.reindex(index=current_return.index)  # 删除当天return为空的样本
        current_weight = weight[date].reindex(index = current_return.index)

        data = pd.concat([current_return,risk_columns,current_industry,current_weight],axis = 1,join="inner")
        data.columns = ["return"]+ factor_list + ["weight"]              # 原始data的column名均为日期，现改为真实名字
        # data = data.dropna(how = "any",axis = 0)                       # Drop all NaN

        # 输出日志
        industry_observation_num = list(data.iloc[:,10:-1].sum(0))
        industry_tuple = [(a,b) for a,b in zip(industry_name_list,industry_observation_num)]
        log.debug(f"Observation Num for{file_name}:{date} is {data.shape[0]}")
        log.debug(f"Industry observation number: {industry_tuple}")
        data = fill_by_category(data, [[0,10],42], [10,42])

        data_y = data.iloc[:,0]
        data_x = data.iloc[:,1:-1]
        # print("x shape:",data_x.shape)
        # print("y shape:",data_y.shape)
        result = base_class.regression(data_x, data_y, data.iloc[:,-1])
        t_df[date] = result.tvalues
        coef_df[date] = result.params/period
        std_error_df[date] = result.bse/period

        """获得因子自稳定系数"""
        try:
            stock_list = set(list(data_t_1.index) + list(data_x.index))       # 只选择今天和昨天都有的股票
            data_x_t_1 = data_t_1.reindex(index=stock_list)
            data_x_t = data.iloc[:,1:10].reindex(index=stock_list)
            weight_t_1 = weight_t_1.reindex(index=stock_list)
            weight_array = pd.DataFrame(weight_t_1).values                    # 将Series转化为DataFrame，然后取array用于广播
            log.debug(f"data t-1:{data_x_t_1.shape};data t:{data_x_t.shape}; weight:{weight_array.shape}")

            # 算分子
            cov_array = (data_x_t - data_x_t.mean(0)).values * (data_x_t_1 - data_x_t_1.mean(0)).values
            numerator = np.nansum(cov_array * weight_array,axis=0)

            # 算分母
            var_t_array = ((data_x_t - data_x_t.mean(0)) ** 2).values * weight_array
            var_t_1_array = ((data_x_t_1 - data_x_t_1.mean(0)) ** 2).values * weight_array
            denominator = (np.nansum(var_t_array,axis=0) * np.nansum(var_t_1_array,axis=0)) ** 0.5

            factor_sc = numerator / denominator                      # 单期因子自稳定系数array
            factor_scs.append(list(factor_sc))

        except NameError :
            pass

        data_t_1 = data.iloc[:,1:10]            # 更新数据：上一期的风险因子载荷
        weight_t_1 = data.iloc[:,-1]            # 更新数据：上一期的权重

    print(factor_scs)
    factor_stability_coef.iloc[:,:] = np.nanmean(np.array(factor_scs),axis=0)       # 将得到的值赋给之前建立的因子自稳定系数空壳
    t_df.to_csv("./output/t_df_" + file_name + ".csv",encoding="gbk")
    coef_df.to_csv("./output/coef_df_" + file_name + ".csv",encoding="gbk")

def get_statistics(file_name):
    indicator_obj = validity.validity(len(risk_factor_list + industry_name_list) + 1, coef_df, t_df, std_error_df,
                                      start_date_time, end_date_time)  # 调用validity文件算出各种参数
    indicators_name = ['|t|_mean', '|t|>2 %', 'annual_return', 'annual_vol',
                       'sharp_ratio',"factor_stability_coef"]  # ,'corr_HS300'] # 顺序和报告一致，hs300目前没有数据所以算不出来

    Rf = 0.02

    sharpe_ratio = (indicator_obj.annual_return() - Rf) / indicator_obj.annual_volatility()  # 算夏普比率
    indicators = [indicator_obj.t_mean_abs(), indicator_obj.t_ratio(), indicator_obj.annual_return(),
                  indicator_obj.annual_volatility(), sharpe_ratio,factor_stability_coef.T]  # 算指标，indicators是一个list，list里的每个元素都是一列df

    summary = pd.concat(indicators, axis=1,join="outer")       # 把上面list里的df合起来
    summary.index = factor_list                   # 把index改成factor_list
    summary.columns = indicators_name
    summary.to_csv("./output/summary_" + file_name + ".csv", encoding="gbk")
    log.info("Getting factor return statistics is Done")

"""#####################画图#####################"""
# 获得各个因子收益率的净值曲线
def coefplot(coef_df,file_name):
    """画出各个因子的收益率曲线"""
    plot_data = coef_df.T                                           # index - 时间；column - 因子
    index_list = list(plot_data.index)                              # 时间列表

    l = Line(width=1500, height=600)

    for i in range(len(factor_list)):
        plotdata_column = (plot_data.iloc[:, i] + 1).cumprod(0)
        l.add('%s' % factor_list[i], index_list, plotdata_column,
              is_fill=False, line_opacity=0.8,
              is_smooth=True, is_datazoom_show=True,
              datazoom_type='both', yaxis_min='dataMin')
    # grid = Grid(height=600, width=1200)
    # grid.add(l, grid_top="12%", grid_left="10%")
    # grid.render(path='/Users/yutong/Desktop/atest.html')

    l.render(path="./output/" + file_name + ".html")    # 保存曲线图
    log.info("Drawing line is Done")
# coefplot(coef_df)


"""#####################获得模型结果#####################"""


"""组合：选择一个组合，注销其他组合(更改文件名)"""
# 月度 - WLS - 多因子

regression = "TRY FSCS"
log.info(f"<<<<<<<<<<<<<<<<<<<<<<<<<<<New regression : {regression}>>>>>>>>>>>>>>>>>>>>>>>")
multi_risk_test(Return_month,regression,21)
get_statistics(regression)
coefplot(coef_df,regression)

# 日度 - WLS - 多因子

# regression = "NEW_T+2_MULTI"
# log.info(f"<<<<<<<<<<<<<<<<<<<<<<<<<<<New regression : {regression}>>>>>>>>>>>>>>>>>>>>>>>")
# multi_risk_test(Return,regression,2)
# get_statistics(regression)
# coefplot(coef_df,regression)

# # 月度 - WLS - 单因子
# regression = "mon_single_fill_mean"
# log.info(f"<<<<<<<<<<<<<<<<<<<<<<<<<<<New regression : {regression}>>>>>>>>>>>>>>>>>>>>>>>")
# single_risk_test(Return_month,regression,21)
# get_statistics(regression)
#
# # 日度 - WLS - 单因子
#
# regression = "t+2_single_fill_mean"
# log.info(f"<<<<<<<<<<<<<<<<<<<<<<<<<<<New regression : {regression}>>>>>>>>>>>>>>>>>>>>>>>")
# single_risk_test(Return,regression,2)
# get_statistics(regression)

# 日度 - WLS - 单因子+行业

# regression = "T+2_SINGLE+INDUSTRY"
# log.info(f"<<<<<<<<<<<<<<<<<<<<<<<<<<<New regression : {regression}>>>>>>>>>>>>>>>>>>>>>>>")
# single_risk_industry_test(Return,regression,2)
# get_statistics(regression)
# coefplot(coef_df,regression)

# 月度 - WLS - 单因子+行业

# regression = "MON_SINGLE+INDUSTRY"
# log.info(f"<<<<<<<<<<<<<<<<<<<<<<<<<<<New regression : {regression}>>>>>>>>>>>>>>>>>>>>>>>")
# single_risk_industry_test(Return_month,regression,21)
# get_statistics(regression)
# coefplot(coef_df,regression)
