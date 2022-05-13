import math
import numpy as np
from numpy import *
import pandas as pd
import statsmodels.api as sm
from numpy import nan as NaN
from pandas.core.frame import DataFrame
import os
import cx_Oracle
import datetime
import time
import tushare as ts

#start=time.time()

'''
dir_path=input('Enter dir_path:') #read calculation_based data
des_path=input('Enter des_path:') #read risk-factor store
dir_rfhist_path=input('Enter dir_rfhist_path:') #read risk_factor store
startdate=input('Enter startdate（计算范围内最早的交易日）:') #has to be tradedays
enddate=input('Enter enddate（计算范围内最晚的交易日）:') #has to be tradedays
'''

#dir_path='/usr/datashare/fun_pq_factors'
#des_path='/usr/datashare/risk_factors'
#dir_rfhist_path='/usr/datashare/risk_factors'


dir_path='/usr/intern/zzzz'
des_path='/usr/intern/zzzz/risk_factors'
dir_rfhist_path='/usr/intern/zzzz/risk_factors'

#startdate='20180901' #has to be tradedays
#enddate='20181101' #has to be tradedays
#startdate=input('Enter startdate:')
#enddate=input('Enter enddate:')

###########################日期判断#################################

#下面的两个函数判断某天是不是交易日以及取输入日期的上一个交易日
def is_trade_date(day): #判断今天是不是交易日
    trade_date_list = ts.trade_cal().set_index("calendarDate") #取一个交易日dataframe
    day=pd.to_datetime(day).date()
    day=day.strftime('%Y-%m-%d')
    key=trade_date_list.loc[day,"isOpen"] #判断是不是交易日，返回true or false
    return key


def last_trade_date(day): #返回上一个交易日
    day_pointer=pd.to_datetime(day).date()+datetime.timedelta(-1) 
    while is_trade_date(day_pointer)==0:
        day_pointer=day_pointer+datetime.timedelta(-1)
    last_trade_date=day_pointer.strftime('%Y%m%d')
    return last_trade_date

def next_trade_date(day): #返回下一个交易日
    day_pointer = pd.to_datetime(day).date() + datetime.timedelta(1)
    while is_trade_date(day_pointer) == 0:
        day_pointer = day_pointer + datetime.timedelta(1)
    next_trade_date = day_pointer.strftime('%Y%m%d')
    return next_trade_date


###########################常用函数#################################
def change_format(df, method="transfer_method"):
    # 作用：改变column日期数据的格式
    # 输入：method，"int2date" - int转为datetime格式；"date2int" - datetime转为int格式
    test_columns = ["EST_DT", "EST_REPORT_DT", "ANN_DT", "REPORT_PERIOD", "REPORTING_PERIOD","S_STM_PREDICT_ISSUINGDATE","S_STM_ACTUAL_ISSUINGDATE","S_STM_CORRECT_ISSUINGDATE"]
    if method == "int2date":
        for column in test_columns:
            if column in df.columns:
                df[column] = df[column].astype(str)
                df[column] = pd.to_datetime(df[column], format='%Y%m%d')
    if method == "date2str":
        for column in test_columns:
            if column in df.columns:
                df[column] = df[column].apply(lambda x: x.strftime("%Y%m%d"))
    return df

def Q3_announce_date():
    # 获得Q3财报发布日
    '''
     输出：DataFrame: 列：S_INFO_WINDCODE,year,ANN_DT
    '''
    season3 = pd.read_csv(dir_path+'/q3_date.csv')
    season3 = change_format(season3,"int2date")

    index = ((season3["REPORT_PERIOD"].dt.month == 9) & (season3["REPORT_PERIOD"].dt.day == 30))
    season3 = season3[index]
    season3["year"] = season3["REPORT_PERIOD"].dt.year
    final = season3.drop(["TOT_ASSETS", "REPORT_PERIOD"], axis=1)
    return final

def combine_factor(factor_list, weight_list):
    """
    :param:factor_list:引起df list,weight_list: 权重list
    :return: 组合后的因子
    """
    weight_df_list = [~(df.isna()) * weight for df, weight in zip(factor_list, weight_list)]
    factor_list = [factor.fillna(1) for factor in factor_list]  # fill na for factor DataFrame,防止有缺失值的自动忽略
    # 对weight_df_list内的元素求和，sum()函数无法运行，使用for循环
    weight_sum = pd.DataFrame(0,index = factor_list[0].index,columns = factor_list[0].columns)  # 求和矩阵都设置为0，防止nan是不相加
    for item in weight_df_list:
        weight_sum = weight_sum + item
    weight_sum = weight_sum.replace(to_replace=0, value=np.nan)  # 权重之和为0表明三个都是缺失值，则该值赋值nan
    factor_weight = [factor * weight for factor, weight in zip(factor_list, weight_df_list)]
    # 对factor_weight内的元素求和，sum()函数无法运行，使用for循环
    factor_weight_sum = pd.DataFrame(0,index = factor_list[0].index,columns = factor_list[0].columns)
    for item in factor_weight:
        factor_weight_sum = factor_weight_sum + item
    combined_factor = factor_weight_sum / weight_sum     # 权重之和为0的记为Nan
    combined_factor = combined_factor.fillna(method = "pad",axis = 1, limit = 250)   # 向后补充Nan,即补充factor_list都是缺失值的数据

    return combined_factor



def HistData(filename):             #处理历史数据,当不存在历史数据时，让它变成空的dataframe
    try:
        hist_data=pd.read_csv(dir_rfhist_path+'/'+filename).set_index('S_INFO_WINDCODE')
    except FileNotFoundError:
        hist_data=pd.DataFrame()
    return hist_data
'''
if startdate==enddate:
    startdate=last_trade_date(startdate)
    enddate = last_trade_date(enddate)
else:
    if is_trade_date(startdate)==0:
        startdate=next_trade_date(startdate)
    if is_trade_date(enddate)==0:
        enddate=last_trade_date(enddate)
    if enddate==datetime.date.today().strftime('%Y%m%d'):
        enddate = last_trade_date(enddate)
'''

#若每天跑：
'''
def get_time():
    # 判断今日是不是周一
    to_weekday = datetime.date.today().weekday()
    # 获取今天的日期
    today = datetime.date.today()
    today_format = today.strftime('%Y%m%d')
    # 周一，获取上周五
    if to_weekday == 0:
        # 获取昨天的日期
        yesterday = datetime.date.today() + datetime.timedelta(-3)
        yesterday_format = yesterday.strftime('%Y%m%d')
    # 其他工作日，正常
    else:
        # 获取昨天的日期
        yesterday = datetime.date.today() + datetime.timedelta(-1)
        yesterday_format = yesterday.strftime('%Y%m%d')
    startdate=enddate=yesterday_format
    return startdate,enddate
#startdate,enddate=get_time()
'''

#reindex至3637
#这个在我的pycharm里面会神奇地卡住，用jupyter很顺畅；
'''
sl=pd.read('/Volumes/rachaelzq/test_hist/stock_list.csv')
stock_list=list(sl['S_INFO_WINDCODE'])
def reindex3637(target):
    df=pd.read_csv(dir_path+'/'+target+'.csv').set_index('S_INFO_WINDCODE')
    df=df.reindex(stock_list)
    df.to_csv(dir_path+'/'+target+'.csv')
    print(target,len(df))
    return target,len(df)
'''

###########################标准化#################################

#normalization of risk factors
def normalization(df): #去极值+标准化函数
    df_list=df.columns
    df_length=len(df_list)
    for i in range(0,df_length):
        a = np.abs(df.iloc[:,i] - df.iloc[:,i].median())         # 每列的中位数以及每个数与它的差值
        diff = a.median()                                        # 差值的中位数

        ##########need better way to fix inf for alpha114 on 20150708
        if (diff == np.inf):
            aa = a.replace(np.inf, a[a != np.inf].max())
            diff = aa.median()
        ##########need better way to fix inf for alpha114 on 20150708

        ##########need better way to fix -inf
        if (diff == -np.inf):
            aa = a.replace(-np.inf, a[a != -np.inf].min())
            diff = aa.median()
        ##########need better way to fix -inf
        if diff == 0:                                            # 超过一半等于中位数时 -> diff =0
            diff = df.iloc[:,i].mad()                            # 使用平均绝对离差作为diff
        if np.isnan(diff) == True:
            continue

        maxrange = df.iloc[:,i].median() +4 * diff                 # 上限为中位数+4倍的diff
        minrange = df.iloc[:,i].median() -4 * diff                 # 下限为中位数-4倍的diff
        df.iloc[:,i] = df.iloc[:,i].clip(minrange,maxrange)      # 去极值，把所有数据控制在上下限范围内，大于上限的等于上限，小于下限的等于下限

        df.iloc[:,i] = (df.iloc[:,i]-np.mean(df.iloc[:,i])) / np.std(df.iloc[:,i]) # 标准化
    return df
"""
#circulation_value:
#as weight,no need to be normalized
def getStock_Circulation_value(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,trade_dt,s_dq_mv from wind.AShareEODDerivativeIndicator where trade_dt>=%s AND trade_dt<=%s " %(x_str,y_str)
    d = pd.read_sql(sql,db)
    d = d.set_index(['TRADE_DT','S_INFO_WINDCODE'])
    d =d.stack(level = 0)
    d = d.unstack(level = 0)
    db.close()
    d.index = d.index.droplevel(1)
    d_hist = pd.read_csv(dir_rfhist_path + '/weight.csv').set_index('S_INFO_WINDCODE')
    d1=pd.concat([d_hist,d],axis=1,sort=True)
    d1.index.name='S_INFO_WINDCODE'
    d1.to_csv(des_path+'/weight.csv')
    return 'circulation_value done'
getStock_Circulation_value(startdate,enddate)
print('circulation value done')
"""
"""
#industry:
#has to be run each time as new stocks enter
#需要每天更新，因为会退市，但用到未来的信息
def get_industry_new():
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl",encoding='gbk')
    sql = '''select A.s_info_windcode,A.sw_ind_code,B.industriesname SECTOR from wind.AShareSWIndustriesClass A,wind.AShareIndustriesCode B where A.cur_sign = '1' And B.used = '1' and B.levelnum = '3' and (substr(A.SW_IND_CODE,1,4) = '6134' or substr(A.SW_IND_CODE,1,4) = '6112' or substr(A.SW_IND_CODE,1,4) = '6118') and (substr(A.SW_IND_CODE,1,6) = substr(B.industriescode,1,6))
    union select A.s_info_windcode,A.sw_ind_code,B.industriesname SECTOR from wind.AShareSWIndustriesClass A,wind.AShareIndustriesCode B where A.cur_sign = '1' And B.used = '1' and B.levelnum = '2' and (substr(A.SW_IND_CODE,1,4) <> '6134' and substr(A.SW_IND_CODE,1,4) <> '6112' and substr(A.SW_IND_CODE,1,4) <> '6118') and (substr(A.SW_IND_CODE,1,4) = substr(B.industriescode,1,4))'''
    d = pd.read_sql(sql,db)
    industry_LevelDict = {
        '交通运输': 1,
        '休闲服务': 2,
        '传媒': 3,
        '公用事业': 4,
        '农林牧渔': 5,
        '化工': 6,
        '医药生物': 7,
        '商业贸易': 8,
        '国防军工': 9,
        '家用电器': 10,
        '建筑材料': 11,
        '建筑装饰': 12,
        '房地产开发Ⅱ': 13,
        '园区开发Ⅱ':14,
        '有色金属': 15,
        '机械设备': 16,
        '汽车': 17,
        '电子': 18,
        '电气设备': 19,
        '纺织服装': 20,
        '综合': 21,
        '计算机': 22,
        '轻工制造': 23,
        '通信': 24,
        '采掘': 25,
        '钢铁': 26,
        '银行': 27,
        '保险Ⅱ': 28,
        '证券Ⅱ':29,
        '多元金融Ⅱ':30,
        '食品加工': 31,
        '饮料制造':32
    }
    industry = d.drop(columns = 'SW_IND_CODE')
    industry = industry.set_index('S_INFO_WINDCODE')
    industry['SECTOR'] = industry['SECTOR'].map(industry_LevelDict)
    industry['SECTOR'].drop_duplicates()
    dummies = pd.get_dummies(industry, columns=['SECTOR'], prefix=['Industry'], prefix_sep="_", dummy_na=False,
                             drop_first=False)
    dummies.to_csv(des_path+'/industry.csv')
    return 'industry done'
get_industry_new()
print('industry done')
"""

###########################因子处理#################################

#size
def getStyle_size(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,trade_dt,s_val_mv from wind.AShareEODDerivativeIndicator where trade_dt>=%s AND trade_dt<=%s " % (
    x_str, y_str)
    df = pd.read_sql(sql, db)
    d = pd.DataFrame(df)
    d = d.set_index(['TRADE_DT', 'S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    temp=pd.DataFrame(index=stock_list,columns=return_timelist)
    temp.loc[:,:]=d.loc[:,:]
    d=np.log(temp)     #按照五十七定义，应先对市值取对数，再做标准化
    d=normalization(d)
    d_hist=HistData('size.csv')
    d1=pd.concat([d_hist,d],axis=1,sort=True)
    d1.index.name='S_INFO_WINDCODE'
    d1.to_csv(des_path + '/size.csv')
    dd=pd.DataFrame(index=stock_list,columns=return_timelist)#只需要取更新时间段的数据
    dd.loc[:,:]=temp.loc[:,:] #原来为dd.loc[:,:]=d.loc[:,:],改完之后返回的dd是没有标准化的原始市值
    dd.index.name='S_INFO_WINDCODE'
    return dd

#ME=pd.read_csv(dir_path + '/size.csv',error_bad_lines=False).set_index('S_INFO_WINDCODE')
#pb_value
def getStyle_value(x,y): #获得value，但应数据库中没有（总权益/当前市值）用净资产代替总权益
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,trade_dt,s_val_pb_new from wind.AShareEODDerivativeIndicator where trade_dt>=%s AND trade_dt<=%s " % (
    x_str, y_str)
    df = pd.read_sql(sql, db)
    d = pd.DataFrame(df)
    d = d.set_index(['TRADE_DT', 'S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    temp=pd.DataFrame(index=stock_list,columns=return_timelist)
    temp.loc[:,:]=d.loc[:,:]
    d=temp
    d1=1/d                              # value定义为（总权益除以当前市值），因此需要倒数一下
    d1 = normalization(d1)
    d_hist=HistData('value.csv')
    d2=pd.concat([d_hist,d1],axis=1,sort=True)
    d2.index.name='S_INFO_WINDCODE'
    d2.to_csv(des_path + '/value.csv')
    return

# momentum
def get_momentum():
    # delta = math.pow(0.5, 1 / 120)  # return 0.5的1/126次方.根据研报暂时把参数调成一致126改为120，504改成500
    T = 500  # 数据长度要足够长，需要close和return有从startdate的两年前开始直到startdate这么多的数据，否则算momentum会报错
    L = 21

    R = np.log(Return + 1)
    new_mmt = pd.DataFrame(index=R.index,columns=R.columns)
    for i in range(startindex,endindex+1):
        new_mmt.iloc[:,i] = R.iloc[:,i-(T+L):i-L].ewm(halflife=120,axis=1).mean().iloc[:,-1]
    new_mmt = new_mmt.iloc[:,-(endindex-startindex):]
    #
    #
    # weight = []  # 长度为500
    # for j in range(1, T + 1):
    #     weight.append(math.pow(delta, T + 1 - j))
    # mmt = dict()
    # for i in range(0, len(stock_list)):
    #     mmt[stock_list[i]] = dict()
    #     #print(i)
    #     for t in range(startindex, endindex + 1, 1):
    #         r = R.iloc[i, (t - T - L):(t - L)]  # 长度为500
    #         mmt[stock_list[i]][timelist[t]] = sum(weight * r)   # 注意，如果数据从数据不足21会报错，所以risk factor的时间应该晚于data的时间
    #
    # mmt1 = pd.DataFrame(mmt).T
    mmt1 = new_mmt.replace({0: np.nan})                      # TODO:为什么要把0替换为nan？
    #这个地方，由于前面没有值的会都为0，所以这里也会被fill成0，这样在标准化的时候就会出问题
    #mmt1.index.name='S_INFO_WINDCODE'
    mmt1=normalization(mmt1)
    mmt_hist=HistData('momentum.csv')
    mmt2=pd.concat([mmt_hist,mmt1],axis=1,sort=True)
    mmt2.index.name='S_INFO_WINDCODE'
    mmt2.to_csv(des_path+'/momentum.csv')
    return

# leverage
def get_leverage():
    LD = pd.read_csv(dir_path + '/Longterm_Liability.csv').set_index('S_INFO_WINDCODE').loc[:,startdate:enddate]
    TD = pd.read_csv(dir_path + '/Total_debt.csv').set_index('S_INFO_WINDCODE').loc[:,startdate:enddate]
    TA = pd.read_csv(dir_path + '/Total_Asset.csv').set_index('S_INFO_WINDCODE').loc[:,startdate:enddate]
    BE = pd.read_csv(dir_path + '/Book_Equity.csv').set_index('S_INFO_WINDCODE').loc[:,startdate:enddate]
    #for x in [LD, TD, TA, BE]:
        #x = x.set_index('S_INFO_WINDCODE', inplace=True)
        #x = x[:,startindex:endindex+1] #只切取需要更新的时间段,一定要是一个长度的
        #x = x.loc[:,startdate:enddate+1]
    LD = LD.fillna(0)
    #ME_new = pd.DataFrame(index=ME.index, columns=timelist)
    #ME_new.loc[:, :] = ME.loc[:, :]
    #ME = ME_new * 10000
    ME_new = ME * 10000 # ME的单位是万元，所以乘以10000
    MLEV = (ME_new + LD) / ME_new
    DTOA = TD / TA
    BLEV = (BE + LD) / BE

    factor_list = [MLEV,DTOA,BLEV]
    weight_list = [0.38,0.35,0.27]
    lev = combine_factor(factor_list = factor_list,weight_list=weight_list)

    # lev = 0.38 * MLEV + 0.35 * DTOA + 0.27 * BLEV

    lev = normalization(lev)
    lev_hist=HistData('leverage.csv')
    lev_new=pd.concat([lev_hist,lev],axis=1,sort=True)
    lev_new.index.name='S_INFO_WINDCODE'
    lev_new.to_csv(des_path + '/leverage.csv')
    return 'leverage done'

#liquidity
def get_liquidity():#应该需要重写，T1求和时应为3*21，T2求和应为21*12
    volume = pd.read_csv(dir_path + '/volume.csv', index_col=0)
    CCS = pd.read_csv(dir_path + '/circulation_capitalStock.csv', index_col=0)

    rolling_days = 21
    hist_data_window = 12 * 21
    VC = volume.iloc[:,startindex-hist_data_window:endindex+1] / CCS.iloc[:,startindex-hist_data_window:endindex+1]
    STOM = np.log(VC.rolling(window=rolling_days,axis = 1).sum())
    STOQ = np.log(VC.rolling(window=rolling_days*3,axis=1).sum()/3)
    STOA = np.log(VC.rolling(window=rolling_days*12,axis=1).sum()/12)

    factor_list = [STOM,STOQ,STOA]
    weight_list = [0.35,0.35,0.30]
    liquidity = combine_factor(factor_list=factor_list,weight_list=weight_list)

    # liquidity = 0.35 * STOM + 0.35 * STOQ + 0.30 * STOA

    liquidity = liquidity.iloc[:, startindex-endindex:] #只取更新的片段
    liquidity = normalization(liquidity)
    liquidity_hist = HistData('liquidity.csv')
    liquidity_new = pd.concat([liquidity_hist,liquidity],axis=1,sort=True)
    liquidity_new.index.name = 'S_INFO_WINDCODE'
    liquidity_new.to_csv(des_path + '/liquidity.csv')
    return 'liquidity done'

class get_alltimelist():#函数没问题，可以再优化
    """取区间日期模块，主要传入两个参数:
    Quriqi('20180601','20180608').suanriqi()  将会返回list类型
    此类用的模块为datetime
    """
    def __init__(self, qishi, jiezhi):
        self.start = qishi
        self.end = jiezhi
        self.suanriqi()
    def suanriqi(self):
        datestart = datetime.datetime.strptime(self.start, '%Y%m%d')
        dateend = datetime.datetime.strptime(self.end, '%Y%m%d')
        riqi_list = []
        riqi_list.append(datestart.strftime('%Y%m%d'))
        while datestart < dateend:
            datestart += datetime.timedelta(days=1)
            qu = datestart.strftime('%Y%m%d')
            riqi_list.append(qu)
        return riqi_list

#growth(SGRO3)

def get_growth_factor(factor_name):
    # 用于获得SGRO和EGRO因子
    # factor name = "SGRO" or "EGRO"
    if factor_name == "SGRO":
        data = pd.read_csv(dir_path + '/operation_revenue_ss.csv')
        data_column = "TOT_OPER_REV"
        ttm_name = "operation_revenue_ttm"
    elif factor_name == "EGRO":
        data = pd.read_csv(dir_path + '/net_profit_ss.csv')
        data_column = "NET_PROFIT_INCL_MIN_INT_INC"
        ttm_name = "net_profit_ttm"
    else: pass
    # print('step1 data: ', data)
    # 要求，传入带有REPORT_PERIOD的DataFrame

    # 分离ann_dt和report period
    data["ANN_DT"] = data["ANN_DT"].astype(str)
    data = data.set_index(["S_INFO_WINDCODE","REPORT_PERIOD"])
    ann_dt = pd.DataFrame(data["ANN_DT"])
    data = data.drop(["ANN_DT"],axis = 1).unstack(1)
    # print('data: ',data)

    # 求出五年简单增长率
    data_ttm = data.rolling(window=4,axis=1).sum()
    data_sum = (data_ttm - data_ttm.shift(periods=20,axis=1)) / abs(data_ttm.shift(periods=20,axis=1))
    data_sum = data_sum / 5
    data_growth = data_sum.stack()
    
    #合并ann_dt与ttm数据
    ttm = ann_dt.copy()
    data_ttm = data_ttm.stack()
    ttm[data_column] = data_ttm[data_column]
    data_ttm = ttm.reset_index()
    # print('ttm: ', ttm)
    data_ttm = data_ttm.drop(["REPORT_PERIOD"],axis = 1)
    data_ttm = data_ttm.set_index(["S_INFO_WINDCODE","ANN_DT"])

    data_ttm = data_ttm.unstack("ANN_DT")
    data_ttm.columns = data_ttm.columns.droplevel(0)

    #填充数据对齐，并保存
    data_ttm = data_ttm.reindex(index = stock_list,columns = all_timelist)
    data_ttm = data_ttm.fillna(method='pad', axis=1, limit=252)
#    data_ttm.to_csv(des_path + "/bbb.csv")
    data_ttm = data_ttm.reindex(index = stock_list,columns = timelist)
    data_ttm.to_csv(dir_path + "/%s.csv"%ttm_name)

    # 合并ann_dt和数据growth
    ann_dt[data_column] = data_growth[data_column]
    grow_rate = ann_dt.reset_index()
    grow_rate = grow_rate.drop(["REPORT_PERIOD"],axis = 1)
    grow_rate = grow_rate.set_index(["S_INFO_WINDCODE","ANN_DT"])
    grow_rate = grow_rate.unstack("ANN_DT")
    grow_rate.columns = grow_rate.columns.droplevel(0)

    # 填充数据
    grow_rate = grow_rate.reindex(index = stock_list,columns = all_timelist)
    grow_rate = grow_rate.fillna(method='pad', axis=1, limit=252)
    grow_rate = grow_rate.reindex(index = stock_list,columns = timelist)

    return grow_rate

def get_EGIB():

    d = pd.read_csv(dir_path+'/est_profit.csv')
    real_profit = pd.read_csv(dir_path + '/net_profit_ttm.csv',index_col = 0)

    sort = d.sort_values(by=["S_INFO_WINDCODE", "EST_DT", "EST_REPORT_DT"])
    drop_duplicated = sort.duplicated(["S_INFO_WINDCODE", "EST_DT"], keep="last")  # 产生布尔值，删除值 = True
    d = sort[drop_duplicated == False]  # 删除掉同一只股票同一天发布的全部预测，保留最后一个

    d = change_format(d,"int2date")

    d['diff'] = d["EST_REPORT_DT"] - d["EST_DT"]
    d['diff'] = d['diff'].apply(lambda x: x.days)
    d = d[d['diff'] > 365]
    d = d[d['diff'] < 730]                                                                                              # 长度是一年到两年的
    d = d.drop_duplicates(['EST_DT'], keep='first')
    d['EST_DT'] = d['EST_DT'].apply(lambda x: str(x.strftime("%Y%m%d")))
    d['EST_REPORT_DT'] = d['EST_REPORT_DT'].apply(lambda x: str(x.strftime("%Y%m%d")))
    d = pd.pivot_table(d, values='NET_PROFIT_AVG', index='S_INFO_WINDCODE', columns='EST_DT')

    d = d.reindex(index = stock_list,columns = all_timelist)
    d = d.fillna(method='pad', axis=1, limit=252)
    est = d.reindex(index = stock_list,columns = timelist)

    growth_rate = (est - real_profit)/(2 *abs(real_profit))    #
    return growth_rate

def get_EGIB_s():

    df = pd.read_csv(dir_path+'/est_profit.csv')
    real_profit = pd.read_csv(dir_path+'/net_profit_ttm.csv',index_col = 0)

    df = change_format(df,"int2date")
    q3_date = Q3_announce_date().set_index(["S_INFO_WINDCODE","year"])  # 需要在data_need_update文件上补充Q3_announce_date所需要的文件
    df["year"] = df["EST_DT"].dt.year
    df = df.set_index(["S_INFO_WINDCODE", "year"])
    df["Q3_ANN_DT"] = q3_date["ANN_DT"]
    df["early_or_later"] = 1
    df["early_or_later"] = df["early_or_later"].where(df["EST_DT"] < df["Q3_ANN_DT"], 0)  # 预测日小于Q3财报日记为1，否则记为0
    df["est_report_year_diff"] = df["EST_REPORT_DT"].dt.year - df["EST_DT"].dt.year
    df["drop_indicator"] = df["early_or_later"] + df["est_report_year_diff"]  # 删除指示变量，1 -> 保留，否则删除
    df = df[df["drop_indicator"].isin(["1"])]
    df = df.reset_index()
    df = df.drop(["Q3_ANN_DT", "early_or_later", "est_report_year_diff", "drop_indicator", "EST_REPORT_DT", "year"], axis=1)
    df = df.groupby(["S_INFO_WINDCODE", "EST_DT"]).mean()  # 以平均值代替重复值
    df = df.reset_index()
    df = change_format(df,"date2str")
    df = df.set_index(["S_INFO_WINDCODE", "EST_DT"])
    df = df.unstack("EST_DT")
    df.columns = df.columns.droplevel(level=0)

    df = df.reindex(index = stock_list,columns = all_timelist)
    df = df.fillna(method='pad', axis=1, limit=252)
    est = df.reindex(index = stock_list,columns = timelist)

    est_growth_rate = (est - real_profit)/abs(real_profit)

    return est_growth_rate

def get_growth():#其他3个算出来，加上权重,

    # OR=pd.read_csv(dir_path+'/Operation_Revenue.csv')
    # ORS=pd.read_csv(dir_path+'/OR(single season).csv')

    # for x in [OR,ORS]:
    #     x= x.set_index('S_INFO_WINDCODE',inplace=True)

    # 要求，传入带有REPORT_PERIOD的DataFrame

    # 分离ann_dt和report period
    # ORS = ORS.set_index(["S_INFO_WINDCODE","REPORT_PERIOD"])
    # ann_dt = pd.DataFrame(ORS["ANN_DT"])
    # ORS = ORS["TOT_OPER_REV"].unstack(1)
    #
    # # 求出五年简单增长率
    # OR_ttm = ORS.rolling(window=4,axis=1).sum()
    # SGRO = (OR_ttm - OR_ttm.shift(periods=20,axis=1)) / abs(OR_ttm.shift(periods=20,axis=1))
    # SGRO = SGRO / 5
    # SGRO = SGRO.stack()
    #
    # # 合并ann_dt和数据
    # ann_dt["TOT_OPER_REV"] = SGRO["TOT_OPER_REV"]
    # SGRO = ann_dt.reset_index()
    # SGRO = SGRO.drop(["REPORT_PERIOD"],axis = 1)
    # SGRO = SGRO.reset_index(["S_INFO_WINDCODE","ANN_DT"])
    # SGRO = SGRO.stack(1)
    # SGRO.columns = SGRO.columns.droplevel(0)
    #
    # # 填充数据
    # SGRO = SGRO.reindex(index = stock_list,columns = all_timelist)
    # SGRO = SGRO.fillna(method='pad', axis=1, limit=252)
    # SGRO = SGRO.reindex(index = stock_list,columns = timelist)
    
    #若空，权重调为0,其他重新单位化为1
    SGRO = get_growth_factor("SGRO")
    EGRO = get_growth_factor("EGRO")
    EGIB = get_EGIB()
    EGIB_s = get_EGIB_s()

    factor_list = [SGRO,EGRO,EGIB,EGIB_s]
    weight_list = [0.47,0.24,0.18,0.11]
    growth = combine_factor(factor_list,weight_list)
    growth = normalization(growth)
    growth.to_csv(des_path + '/growth.csv')
    return 'growth done'

    # SGRO2=pd.DataFrame(index=SGRO1.index,columns=all_timelist)
    # SGRO2.loc[:,:]=SGRO1.loc[:,:]
    # SGRO2 = SGRO2.fillna( method = 'pad',axis= 1,limit=252)
    # SGRO3=pd.DataFrame(index=SGRO2.index,columns=return_timelist)
    # SGRO3.loc[:,:]=SGRO2.loc[:,:]
    # SGRO3 = normalization(SGRO3)

    # SGRO_hist = HistData('growth.csv')
    # SGRO_new = pd.concat([SGRO_hist,SGRO3],axis=1,sort=True)
    # SGRO_new.index.name='S_INFO_WINDCODE'
    # SGRO_new.to_csv(des_path+'/growth.csv')
    # return



# def EGIB(x,y):
#     x_str = str(x)
#     y_str = str(y)
#     db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
#     sql = "select s_info_windcode,trade_dt,s_dq_high from wind.AShareEODPrices where trade_dt>=%s AND trade_dt<=%s " % (
#     x_str, y_str)
#     df = pd.read_sql(sql, db)
#     d = pd.DataFrame(df)
#EGRO 计算查表AShareFinancialIndicator，字段s_qfa_cgrnetprofit归属于母公司的净利润，用单季度的环比增长率
#EGIB 计算查表AShareConsensusData，字段s_info_windcode,est_dt,est_report_dt,net_profit_avg,num_est_inst。三年后的预测值除以已有的最新年报
#EGIB_S 和上面EGIB一起做，表和字段不变，一年的预测值/最新年报，若有三季报已发布，两年预测值/一年预测值
#权重为0.47 · SGRO + 0.24 · EGRO +0.18 · EGIBS + 0.11 · EGIBS_s


#earning_yield
def get_earning_yield():
    PE = pd.read_csv(dir_path + '/pe.csv', index_col='S_INFO_WINDCODE').loc[:, startdate:enddate]
    P = pd.read_csv(dir_path + '/closeprice_adj.csv', index_col='S_INFO_WINDCODE').loc[:, startdate:enddate]
    eps_est = pd.read_csv(dir_path + '/eps_est.csv', index_col='S_INFO_WINDCODE').loc[:, startdate:enddate]

    #for x in [PE, P, eps_est]: #这样做pycharm会莫名卡；【】取闭区间
    #   x = x.loc[:, startdate:str(int(enddate)+1)] #切取更新的时间段
    #eps_est有问题：test_hist OK,update不对

    EPIBS = eps_est / P #写反了，已修改
    ETOP = 1 / PE
    #CETOP 查表AShareFinancialIndicator，字段s_fa_ocfps，个股经营活动产生现金流/股价P，取到的各季度累计的，减出单季度的，将过去四个加起来
    #方法二查表AShareCashFlow，字段net_cash_flows_oper_act，经营活动产生的现金流量净额/股数 = 个股经营活动产生的现金流量，再进行计算
    #权重为 0.68 · EPIBS + 0.11 · ETOP + 0.21 · CETOP 

    factor_list = [EPIBS,ETOP]
    weight_list = [0.68,0.32]
    earning_yield = combine_factor(factor_list,weight_list)
    # earning_yield = 0.68 * EPIBS + 0.32 * ETOP
    df = earning_yield.reindex(index = stock_list,columns = all_timelist)
    df = df.fillna(method = "pad",axis = 1)
    df = df.reindex(index=stock_list, columns=return_timelist)
    # df.loc[:, :] = earning_yield.loc[:, :]
    df = normalization(df)
    df_hist = HistData('earning_yield.csv')
    df_new = pd.concat ([df_hist,df],axis=1,sort=True)
    df_new.index.name = 'S_INFO_WINDCODE'
    df_new.to_csv(des_path + '/earning_yield.csv' )
    return 'earning_yield done'

##为计算beta做准备
#def getDataReady2(stock, t, f1, f2):  # 现在是t时刻，取t-1及前252天，共252期的数据，t是列号 #hs300,Return,W.
#
#    data1 = f1.iloc[0, (t - 252):t]
#    data2 = f2.iloc[stock, (t - 252):t]
#
#    data2.index = data1.index
#    data = pd.concat([data1, data2], axis=1)
#    data = data.dropna()
#
#    # for j in range(0,2):#标准化结果更不好
#    #    data.iloc[:,j] = (data.iloc[:,j] - mean(data.iloc[:,j])) / std(data.iloc[:,j])
#
#    # print(data)
#    return data

#def regression(x, y, weight):
#    x.astype(float)
#
#    #x_array = x.as_matrix(columns=None)    #as_matrix在后期的版本中不再用了，将其变成value,效果一样
#    #y_array = y.as_matrix(columns=None)
#    x_array=x.values
#    y_array=y.values
#
#    x_array = sm.add_constant(x_array)
#    wls_model = sm.WLS(y_array, x_array, weights=weight)
#    # wls_model = sm.OLS(y_array, x_array)
#    wls_coef = wls_model.fit().params
#    std_residual = np.sqrt(wls_model.fit().mse_resid)
#    # wls_r2 = wls_model.fit().rsquared
#
#    return wls_coef, std_residual

#beta&hsigma
def get_beta_hsigma(): #原意应为对X和Y的序列按250天分别加权平均，再进行OLS，或直接用公式计算
    hs300 = (pd.read_csv(dir_path + '/hs300_return.csv', index_col=0).iloc[:,startindex-260:endindex+1])/100  #百分制收益率
    window_length = 250
    
    def ewma(window):
        window_series = pd.Series(window)
        ewma_result = window_series.ewm(halflife=60).mean()
        return ewma_result[window_length-1]
   
    Return_part = Return.iloc[:,startindex-260:endindex+1]

    #确保return_part和hs300的column保持一致
    hs300 = hs300.reindex(columns = Return_part.columns) # 用return的column调整hs300的column，如果后者更多，则删除
    hs300 = hs300.fillna(method = "pad",axis = 1)  # 如果前者更多，向后填充

    #beta的算法应该是cov（ri，rm）/var（rm），先对x，y的收益率序列平滑，再求beta等价于先求beta，再平滑beta。
#    beta_df = pd.rolling(Return, window=window_length, min_periods=200).corr(hs300) # 前提是Return和hs300的columns一样
#    beta_df_ewma = beta_df.rolling(window=window_length, axis=1).apply(ewma)
    
    hs300_var = np.tile((hs300.rolling(window = window_length, min_periods = 200,axis = 1).var()).values,(len(Return_part.index),1))#hs300_var格式为array格式，维度为3637*time list，方便与下面的cov()矩阵相加减乘除

    cov = (Return_part.T.rolling(window=window_length, min_periods=200).cov(hs300.T['000300.SH']).T)# 前提是Return和hs300的columns一样
    beta_df = cov/hs300_var
    beta_df_ewma = beta_df.rolling(window=window_length, axis=1).apply(ewma,raw = True)
    #beta的残差标准差计算逻辑std(ei) = Var(Return)+Var(beta * hs300)-2((cov(Return,hs300)**2)/Var(hs300)),化简后可得Return_var - ((Cov_returnhs300**2)/hs300_var)
    Return_var = Return_part.rolling(window=window_length, min_periods=200,axis = 1).var() #收益率的波动
    Cov_returnhs300 = (Return_part.T.rolling(window=window_length, min_periods=200).cov(hs300.T['000300.SH'])).T
    volatility = (Return_var - ((Cov_returnhs300**2)/hs300_var))**0.5  #得到的就是回归beta的残差标准差
#    beta = pd.DataFrame(index=stock_list, columns=timelist[regstart:(regend + 1)])
#    volatility = pd.DataFrame(index=stock_list, columns=timelist[regstart:(regend + 1)])
#    #start = datetime.datetime.now()
#    for i in range(0, len(stock_list)):
#        #print(stock_list[i])
#        #istart = datetime.datetime.now()
#        for t in range(regstart, regend + 1, 1):
#            # print(start)
#            # print(t)
#            data = getDataReady2(i, t, hs300, Return)
#            l = len(data.index)
#            if l <= 200:
#                beta.iloc[i, t - regstart] = NaN
#            else:
#                weight = []
#                for j in range(1, l + 1):
#                    weight.append(math.pow(delta, l + 1 - j))
#                datax = data.iloc[:, 0]
#                datay = data.iloc[:, 1]
#                datax = pd.DataFrame(datax, dtype=np.float)
#                coef, std1 = regression(datax, datay, weight)#估计量的公式计算，权重改为weight**2
#                beta.iloc[i, t - regstart] = coef[-1]
#                volatility.iloc[i, t - regstart] = std1
        #print(i)
        #print(datetime.datetime.now() - istart)
    #print(datetime.datetime.now() - start)
    #beta = beta * 100 #乘100干嘛？

#小类因子量纲一值，逻辑相似，小类因子在此先不标准化，最后合成大类因子后再标准化
    beta_df_ewma = beta_df_ewma.loc[:,startdate:enddate]
    beta = normalization(beta_df_ewma)
#    volatility = normalization(volatility)

    beta_hist =HistData('beta.csv')
    #volatility_hist = pd.read_csv (dir_rfhist_path + '/hsigma.csv').set_index('S_INFO_WINDCODE')
    beta_new = pd.concat([beta_hist,beta],axis=1,sort=True)
    #volatility_new = pd.concat([volatility_hist,volatility],axis=1,sort=True)
    beta_new.index.name = 'S_INFO_WINDCODE'
    volatility.index.name = 'S_INFO_WINDCODE'
    #volatility_new.index.name = 'S_INFO_WINDCODE'

    beta_new.to_csv(des_path + '/beta.csv')
    #volatility_new.to_csv(des_path + '/hsigma.csv')
    return volatility


#volatility
def get_volatility():
    delta = math.pow(0.5, 1 / 40)
    T = 250

    weight = []
    for j in range(1, T + 1):
        weight.append(math.pow(delta, T + 1 - j))

    DASTD = dict()

    for i in range(0, len(stock_list)):
        #print(i)
        DASTD[stock_list[i]] = dict()
        for t in range(startindex, endindex + 1, 1):
            r = Return.iloc[i, (t - T):t]
            r = (r - np.mean(r)) * (r - np.mean(r))
            DASTD[stock_list[i]][timelist[t]] = np.sqrt(sum(weight * r) / sum(weight))
    DASTD1 = pd.DataFrame(DASTD).T


    CMRA = dict()
    for i in range(0, len(stock_list)):
        #print(i)
        CMRA[stock_list[i]] = dict()
        for t in range(startindex, endindex + 1, 1):
            temp = Return.iloc[i, (t - 252):t][::-1] #公式为月度收益率，用一年的数据计算,最近的一个月为第一个月，所有计算时先把temp顺序倒一下
            Z = np.log((temp+1).cumprod()[20::21]) #这个为下面那个Z的优化版
#            Z = [x for i,x in enumerate(np.log((temp+1).cumprod())) if i % 21 == 0 ] #Z的数据个数为12个，月度收益率用日度的收益率+1累乘，再取对数，取出每个月第二十一天的收益作为月度收益
            CMRA[stock_list[i]][timelist[t]] = max(Z) -  min(Z)
    CMRA1 = pd.DataFrame(CMRA).T


    factor_list = [DASTD1,CMRA1,HSIGMA]
    weight_list = [0.74,0.16,0.10]
    volatility = combine_factor(factor_list,weight_list)
    # volatility = 0.74 * DASTD1 + 0.16 * CMRA1 + 0.10 * HSIGMA
    volatility = normalization(volatility)

    volatility_hist = HistData('volatility.csv')
    volatility_new = pd.concat([volatility_hist,volatility],axis=1,sort=True)
    volatility_new.index.name='S_INFO_WINDCODE'
    volatility_new.to_csv(des_path + '/volatility.csv')
    return 'volatility done'


###########################函数执行#################################

# startdate='20050107'
startdate = "20180101"
enddate='20180307'
#当dir_rfhist_path中的历史数据不存在时候，只需要更改下面的startdate和enddate,就能算出给定时间段的risk factor数据
#startdate=datetime.date.today().strftime('%Y%m%d')
# enddate=datetime.date.today().strftime('%Y%m%d')

if is_trade_date(enddate)==0:
    pass
else:
    Return=pd.read_csv(dir_path + '/Return.csv',error_bad_lines=False).set_index('S_INFO_WINDCODE') #benchmark, error_bad_lines=False
    
    try:                    	
        size=pd.read_csv(dir_rfhist_path+ '/size.csv',error_bad_lines=False).set_index('S_INFO_WINDCODE')   
        last_update_day=list(size.columns)[-1] #前一次更新后的最后一天，用于确定下次更新开始的时间
        startdate=next_trade_date(last_update_day)
    except FileNotFoundError:
        if is_trade_date(startdate)==0:          
            startdate=next_trade_date(startdate)
        else:
            startdate=startdate
    enddate=last_trade_date(enddate)
    #enddate=last_trade_date(enddate)

    timelist=list(Return.columns) #BIG benchmark,有需要往前计算的因子及其历史数据
    stock_list=list(Return.index) #all stocks
    startindex=timelist.index(startdate) #place of startdate in timelist
    endindex=timelist.index(enddate) #place of enddate in timelist
    return_timelist=timelist[startindex:endindex+1] #NEW benchmark，only需要更新的时间段
    
    ME=getStyle_size(startdate,enddate)
    getStyle_value(startdate,enddate)
    get_momentum()
    get_leverage()
    get_liquidity()
    all_timelist=get_alltimelist(str(timelist[0]),str(timelist[-1])).suanriqi()
    get_growth()
    get_earning_yield()
    
    try:                        
        beta=pd.read_csv(dir_rfhist_path+ '/beta.csv',error_bad_lines=False).set_index('S_INFO_WINDCODE')   
        last_update_day=list(beta.columns)[-1] #前一次更新后的最后一天，用于确定下次更新开始的时间
        startdate=next_trade_date(last_update_day)
    except FileNotFoundError:
        if is_trade_date(startdate)==0:          
            startdate=next_trade_date(startdate)
        else:
            startdate=startdate
    startindex=timelist.index(startdate) #place of startdate in timelist
    #return_timelist=timelist[startindex:endindex+1]
    #print('start')

    HSIGMA=get_beta_hsigma()

    #print('beta done')
    get_volatility()
    #print('vo done')


#end=time.time()
#print((end-start)/60.0)
