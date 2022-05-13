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

dir_path='/usr/datashare/fun_pq_factors'
des_path='/usr/datashare/risk_factors'
dir_rfhist_path='/usr/datashare/risk_factors'
#des_path='/usr/datashare/test_chao/fun_result'

'''
dir_path='/Users/chaowang/Desktop/fun_test'
des_path='/Users/chaowang/Desktop/alpha2'
dir_rfhist_path='/Users/chaowang/Desktop/alpha2'
'''
#startdate='20180901' #has to be tradedays
#enddate='20181101' #has to be tradedays
#startdate=input('Enter startdate:')
#enddate=input('Enter enddate:')

#下面的两个函数判断某天是不是交易日以及取输入日期的上一个交易日
def is_trade_date(day):
    trade_date_list = ts.trade_cal().set_index("calendarDate")
    day=pd.to_datetime(day).date()
    day=day.strftime('%Y-%m-%d')
    key=trade_date_list.loc[day,"isOpen"]
    return key

def last_trade_date(day):
    day_pointer=pd.to_datetime(day).date()+datetime.timedelta(-1)
    while is_trade_date(day_pointer)==0:
        day_pointer=day_pointer+datetime.timedelta(-1)
    last_trade_date=day_pointer.strftime('%Y%m%d')
    return last_trade_date

def next_trade_date(day):
    day_pointer = pd.to_datetime(day).date() + datetime.timedelta(1)
    while is_trade_date(day_pointer) == 0:
        day_pointer = day_pointer + datetime.timedelta(1)
    next_trade_date = day_pointer.strftime('%Y%m%d')
    return next_trade_date

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

#normalization of risk factors
def normalization(df):
    df_list=df.columns
    df_length=len(df_list)
    for i in range(0,df_length):
        a = np.abs(df.iloc[:,i] - df.iloc[:,i].median()) #每列的中位数以及每个数与它的差值
        diff = a.median() #差值的中位数
        if diff == 0 :diff = df.iloc[:,i].mad() #若中位数为0，取差值的平均数
        maxrange = df.iloc[:,i].median() +4*diff #上限为中位数+4倍的diff
        minrange = df.iloc[:,i].median() -4*diff #下限为中位数-4倍的diff
        df.iloc[:,i] = df.iloc[:,i].clip(minrange,maxrange) #去极值，把所有数据控制在上下限范围内，大于上限的等于上限，小于下限的等于下限
        df.iloc[:,i] = (df.iloc[:,i]-np.mean(df.iloc[:,i]))/np.std(df.iloc[:,i]) #标准化
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
    d=temp
    d=normalization(d)
    d_hist=HistData('size.csv')
    d1=pd.concat([d_hist,d],axis=1,sort=True)
    d1.index.name='S_INFO_WINDCODE'
    d1.to_csv(des_path + '/size.csv')
    dd=pd.DataFrame(index=stock_list,columns=return_timelist)#只需要取更新时间段的数据
    dd.loc[:,:]=d.loc[:,:]
    dd.index.name='S_INFO_WINDCODE'
    return dd

#ME=pd.read_csv(dir_path + '/size.csv',error_bad_lines=False).set_index('S_INFO_WINDCODE')
#pb_value
def getStyle_value(x,y):
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
    d1=np.log(d)
    d1 = normalization(d1)
    d_hist=HistData('value.csv')
    d2=pd.concat([d_hist,d1],axis=1,sort=True)
    d2.index.name='S_INFO_WINDCODE'
    d2.to_csv(des_path + '/value.csv')
    return

# momentum
def get_momentum():
    delta = math.pow(0.5, 1 / 126)  # return 0.5的1/126次方
    T = 504  # 数据长度要足够长，需要close和return有从startdate的两年前开始直到startdate这么多的数据，否则算momentum会报错
    L = 21
    R = log(Return + 1)
    weight = []  # 504
    for j in range(1, T + 1):
        weight.append(math.pow(delta, T + 1 - j))
    mmt = dict()
    for i in range(0, len(stock_list)):
        mmt[stock_list[i]] = dict()
        #print(i)
        for t in range(startindex, endindex + 1, 1):
            r = R.iloc[i, (t - T - L):(t - L)]  # 504
            mmt[stock_list[i]][timelist[t]] = sum(weight * r)
    mmt1 = pd.DataFrame(mmt).T
    mmt1 = mmt1.replace({0: np.nan})
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
    ME_new = ME * 10000
    MLEV = (ME_new + LD) / ME_new
    DTOA = TD / TA
    BLEV = (BE + LD) / BE
    lev = 0.38 * MLEV + 0.35 * DTOA + 0.27 * BLEV
    lev = normalization(lev)
    lev_hist=HistData('leverage.csv')
    lev_new=pd.concat([lev_hist,lev],axis=1,sort=True)
    lev_new.index.name='S_INFO_WINDCODE'
    lev_new.to_csv(des_path + '/leverage.csv')
    return 'leverage done'

#liquidity
def get_liquidity():
    volume = pd.read_csv(dir_path + '/volume.csv').set_index('S_INFO_WINDCODE')
    CCS = pd.read_csv(dir_path + '/circulation_capitalStock.csv').set_index('S_INFO_WINDCODE')
    l_startindex=startindex-12 #12=T2
    l_endindex=endindex

    STOM = dict()
    T = 21
    for i in range(0, len(stock_list)):
        #print(i)
        #stomtime = (pd.to_datetime(startdate) + timedelta(-3)).strftime('%Y%m%d')
        #startindexstom = timelist.index(stomtime)
        STOM[stock_list[i]] = dict()
        for t in range(l_startindex, l_endindex + 1, 1):
            data = volume.iloc[i, (t - T):t] / CCS.iloc[i, (t - T):t]
            #print(data)
            STOM[stock_list[i]][timelist[t]] = log(sum(data)) #只截取需要更新的时间片段
    STOM1 = pd.DataFrame(STOM).T
    where_are_inf = np.isinf(STOM1)
    STOM1[where_are_inf] = NaN

    T1 = 3
    STOQ = dict()
    for i in range(0, len(stock_list)):
        STOQ[stock_list[i]] = dict()
        for t in range(l_startindex + T1, l_endindex + 1, 1): #+T1:往后算T1个
            data = exp(STOM1.iloc[i, (t - l_startindex - T1):(t - l_startindex)])
            STOQ[stock_list[i]][timelist[t]] = log(sum(data) / T1)
    STOQ1 = pd.DataFrame(STOQ).T
    where_are_inf = np.isinf(STOQ1)
    STOQ1[where_are_inf] = NaN

    T2 = 12
    STOA = dict()
    for i in range(0, len(stock_list)):
        STOA[stock_list[i]] = dict()
        for t in range(l_startindex + T2, l_endindex + 1, 1): #往后算12个
            data = exp(STOM1.iloc[i, (t - l_startindex - T2):(t - l_startindex)])
            STOA[stock_list[i]][timelist[t]] = log(sum(data) / T2)
    STOA1 = pd.DataFrame(STOA).T
    where_are_inf = np.isinf(STOA1)
    STOA1[where_are_inf] = NaN

    liquidity = 0.35 * STOM1 + 0.35 * STOQ1 + 0.30 * STOA1
    liquidity = liquidity.loc[:, startdate:] #只取更新的片段
    liquidity = normalization(liquidity)
    liquidity_hist = HistData('liquidity.csv')
    liquidity_new = pd.concat([liquidity_hist,liquidity],axis=1,sort=True)
    liquidity_new.index.name = 'S_INFO_WINDCODE'
    liquidity_new.to_csv(des_path + '/liquidity.csv')
    return 'liquidity done'

class get_alltimelist():
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
def get_SGRO3():
    OR=pd.read_csv(dir_path+'/Operation_Revenue.csv')
    ORS=pd.read_csv(dir_path+'/OR(single season).csv')

    for x in [OR,ORS]:
        x= x.set_index('S_INFO_WINDCODE',inplace=True)

    ORR = ORS.rolling(window=4,axis=1).sum()
    ortimelist=list(ORR.columns)

    SGRO=dict()
    for i in range(0,len(stock_list)):
        SGRO[stock_list[i]] = dict()
        for t in range(12,len(ortimelist)):
            if ORR.iloc[i,t]/ORR.iloc[i,t-12]<0:
                SGRO[stock_list[i]][ortimelist[t]]=-math.pow((-ORR.iloc[i,t]/ORR.iloc[i,t-12]),1/3)-1
            else:
                SGRO[stock_list[i]][ortimelist[t]]=math.pow((ORR.iloc[i,t]/ORR.iloc[i,t-12]),1/3)-1
    SGRO1 = pd.DataFrame(SGRO).T
    where_are_inf = np.isinf(SGRO1)
    SGRO1[where_are_inf] = NaN

    SGRO2=pd.DataFrame(index=SGRO1.index,columns=all_timelist)
    SGRO2.loc[:,:]=SGRO1.loc[:,:]
    SGRO2 = SGRO2.fillna( method = 'pad',axis= 1,limit=252)
    SGRO3=pd.DataFrame(index=SGRO2.index,columns=return_timelist)
    SGRO3.loc[:,:]=SGRO2.loc[:,:]
    SGRO3 = normalization(SGRO3)

    SGRO_hist = HistData('growth.csv')
    SGRO_new = pd.concat([SGRO_hist,SGRO3],axis=1,sort=True)
    SGRO_new.index.name='S_INFO_WINDCODE'
    SGRO_new.to_csv(des_path+'/growth.csv')
    return

#earning_yield
def get_earning_yield():
    PE = pd.read_csv(dir_path + '/pe.csv', index_col='S_INFO_WINDCODE').loc[:, startdate:enddate]
    P = pd.read_csv(dir_path + '/closeprice_adj.csv', index_col='S_INFO_WINDCODE').loc[:, startdate:enddate]
    eps_est = pd.read_csv(dir_path + '/eps_est.csv', index_col='S_INFO_WINDCODE').loc[:, startdate:enddate]

    #for x in [PE, P, eps_est]: #这样做pycharm会莫名卡；【】取闭区间
    #   x = x.loc[:, startdate:str(int(enddate)+1)] #切取更新的时间段
    #eps_est有问题：test_hist OK,update不对

    EPIBS = 1 / PE
    ETOP = eps_est / P
    earning_yield = 0.68 * EPIBS + 0.32 * ETOP
    df = pd.DataFrame(index=stock_list, columns=return_timelist)
    df.loc[:, :] = earning_yield.loc[:, :]
    df = normalization(df)
    df_hist = HistData('earning_yield.csv')
    df_new = pd.concat ([df_hist,df],axis=1,sort=True)
    df_new.index.name = 'S_INFO_WINDCODE'
    df_new.to_csv(des_path + '/earning_yield.csv' )
    return 'earning_yield done'

#为计算beta做准备
def getDataReady2(stock, t, f1, f2):  # 现在是t时刻，取t-1及前252天，共252期的数据，t是列号 #hs300,Return,W.

    data1 = f1.iloc[0, (t - 252):t]
    data2 = f2.iloc[stock, (t - 252):t]

    data2.index = data1.index
    data = pd.concat([data1, data2], axis=1)
    data = data.dropna()

    # for j in range(0,2):#标准化结果更不好
    #    data.iloc[:,j] = (data.iloc[:,j] - mean(data.iloc[:,j])) / std(data.iloc[:,j])

    # print(data)
    return data

def regression(x, y, weight):
    x.astype(float)

    #x_array = x.as_matrix(columns=None)    #as_matrix在后期的版本中不再用了，将其变成value,效果一样
    #y_array = y.as_matrix(columns=None)
    x_array=x.values
    y_array=y.values

    x_array = sm.add_constant(x_array)
    wls_model = sm.WLS(y_array, x_array, weights=weight)
    # wls_model = sm.OLS(y_array, x_array)
    wls_coef = wls_model.fit().params
    std_residual = np.sqrt(wls_model.fit().mse_resid)
    # wls_r2 = wls_model.fit().rsquared

    return wls_coef, std_residual

#beta&hsigma
def get_beta_hsigma():
    hs300 = pd.read_csv(dir_path + '/hs300_return.csv').set_index('S_INFO_WINDCODE')
    #hs300.set_index('S_INFO_WINDCODE', inplace=True)
    temp1=pd.DataFrame(index=['000300.SH'], columns=timelist)
    temp1.loc[:,:]=hs300.loc[:,:]
    temp1.index.name = 'S_INFO_WINDCODE'
    hs300=temp1
    delta = math.pow(0.5, 1 / 63)
    alpha = 1 - math.pow(0.5, 1 / 63)

    T = 252
    regstart = timelist.index(startdate)  # regstart=252
    regend = timelist.index(enddate)

    beta = pd.DataFrame(index=stock_list, columns=timelist[regstart:(regend + 1)])
    volatility = pd.DataFrame(index=stock_list, columns=timelist[regstart:(regend + 1)])
    #start = datetime.datetime.now()
    for i in range(0, len(stock_list)):
        #print(stock_list[i])
        #istart = datetime.datetime.now()
        for t in range(regstart, regend + 1, 1):
            # print(start)
            # print(t)
            data = getDataReady2(i, t, hs300, Return)
            l = len(data.index)
            if l <= 200:
                beta.iloc[i, t - regstart] = NaN
            else:
                weight = []
                for j in range(1, l + 1):
                    weight.append(math.pow(delta, l + 1 - j))
                datax = data.iloc[:, 0]
                datay = data.iloc[:, 1]
                datax = pd.DataFrame(datax, dtype=np.float)
                coef, std1 = regression(datax, datay, weight)
                beta.iloc[i, t - regstart] = coef[-1]
                volatility.iloc[i, t - regstart] = std1
        #print(i)
        #print(datetime.datetime.now() - istart)
    #print(datetime.datetime.now() - start)
    beta = beta * 100

    beta = normalization(beta)
    volatility = normalization(volatility)

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
    R = log(1 + Return)

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
            temp = R.iloc[i, (t - 21):(t + 1)]
            Z = temp.cumsum()
            CMRA[stock_list[i]][timelist[t]] = (log(1 + max(Z)) - log(1 + min(Z))) / 21
    CMRA1 = pd.DataFrame(CMRA).T

    volatility = 0.74 * DASTD1 + 0.16 * CMRA1 + 0.10 * HSIGMA
    volatility = normalization(volatility)

    volatility_hist = HistData('volatility.csv')
    volatility_new = pd.concat([volatility_hist,volatility],axis=1,sort=True)
    volatility_new.index.name='S_INFO_WINDCODE'
    volatility_new.to_csv(des_path + '/volatility.csv')
    return 'volatility done'
#startdate='20180602'
#enddate='20181106'
#当dir_rfhist_path中的历史数据不存在时候，只需要更改下面的startdate和enddate,就能算出给定时间段的risk factor数据
startdate=datetime.date.today().strftime('%Y%m%d')
enddate=datetime.date.today().strftime('%Y%m%d')
#以下代码实现：当计算风险因子需要的这些基本面因子全部更新到同一天时候才会更新风险因子，否则数据不全的话，会造成计算的风险因子数据发生错误
hs300_return=pd.read_csv(dir_path+'/hs300_return.csv')
pe=pd.read_csv(dir_path+'/pe.csv')
eps_est=pd.read_csv(dir_path+'/eps_est.csv')
Return=pd.read_csv(dir_path+'/Return.csv')
circulation_capitalStock=pd.read_csv(dir_path+'/circulation_capitalStock.csv')
Longterm_Liability=pd.read_csv(dir_path+'/Longterm_Liability.csv')
Total_debt=pd.read_csv(dir_path+'/Total_debt.csv')
Total_Asset=pd.read_csv(dir_path+'/Total_Asset.csv')
Book_Equity=pd.read_csv(dir_path+'/Book_Equity.csv')
#找到上面这些因子最后的更新日期
last_hs300_return=list(hs300_return.columns)[-1]
last_pe=list(pe.columns)[-1]
last_eps_est=list(eps_est.columns)[-1]
last_Return=list(Return.columns)[-1]
last_circulation_capitalStock=list(circulation_capitalStock.columns)[-1]
last_Longterm_Liability=list(Longterm_Liability.columns)[-1]
last_Total_debt=list(Total_debt.columns)[-1]
last_Total_Asset=list(Total_Asset.columns)[-1]
last_Book_Equity=list(Book_Equity.columns)[-1]
temp=last_hs300_return==last_pe==last_eps_est==last_Return==last_circulation_capitalStock==last_Longterm_Liability==last_Total_debt==last_Total_Asset==last_Book_Equity

if is_trade_date(enddate)==0:
    pass
else:
    if temp:
        enddate=last_trade_date(enddate)
        Return=pd.read_csv(dir_path + '/Return.csv',error_bad_lines=False).set_index('S_INFO_WINDCODE') #benchmark, error_bad_lines=False    
        timelist=list(Return.columns) #BIG benchmark,有需要往前计算的因子及其历史数据
        stock_list=list(Return.index) #all stocks
        try:                    	
            size=pd.read_csv(dir_rfhist_path+ '/size.csv',error_bad_lines=False).set_index('S_INFO_WINDCODE')   
            last_update_day=list(size.columns)[-1] #前一次更新后的最后一天，用于确定下次更新开始的时间
            if last_update_day>=enddate:
                pass
            else:
                startdate=next_trade_date(last_update_day)
                startindex=timelist.index(startdate) #place of startdate in timelist
                endindex=timelist.index(enddate) #place of enddate in timelist
                return_timelist=timelist[startindex:endindex+1] #NEW benchmark，only需要更新的时间段    
                getStyle_size(startdate,enddate)
        except FileNotFoundError:
            if is_trade_date(startdate)==0:          
                startdate=next_trade_date(startdate)
            else:
                startdate=startdate
            startindex=timelist.index(startdate) #place of startdate in timelist
            endindex=timelist.index(enddate) #place of enddate in timelist
            return_timelist=timelist[startindex:endindex+1] #NEW benchmark，only需要更新的时间段    
            getStyle_size(startdate,enddate)
        
        #enddate=last_trade_date(enddate)
        '''    
        startindex=timelist.index(startdate) #place of startdate in timelist
        endindex=timelist.index(enddate) #place of enddate in timelist
        return_timelist=timelist[startindex:endindex+1] #NEW benchmark，only需要更新的时间段    
        ME=getStyle_size(startdate,enddate)
        ME=pd.read_csv(dir_rfhist_path+ '/size.csv',error_bad_lines=False).set_index('S_INFO_WINDCODE')
        '''
        
        try:                        
            value=pd.read_csv(dir_rfhist_path+ '/value.csv',error_bad_lines=False).set_index('S_INFO_WINDCODE')   
            last_update_day=list(value.columns)[-1] #前一次更新后的最后一天，用于确定下次更新开始的时间
            if last_update_day>=enddate:
                pass
            else:
                startdate=next_trade_date(last_update_day)
                startindex=timelist.index(startdate) #place of startdate in timelist
                endindex=timelist.index(enddate) #place of enddate in timelist
                return_timelist=timelist[startindex:endindex+1]
                getStyle_value(startdate,enddate)
        except FileNotFoundError:
            if is_trade_date(startdate)==0:          
                startdate=next_trade_date(startdate)
            else:
                startdate=startdate
            startindex=timelist.index(startdate) #place of startdate in timelist
            endindex=timelist.index(enddate) #place of enddate in timelist
            return_timelist=timelist[startindex:endindex+1]
            getStyle_value(startdate,enddate)
        
        try:
            momentum=pd.read_csv(dir_rfhist_path+ '/momentum.csv',error_bad_lines=False).set_index('S_INFO_WINDCODE')
            last_update_day=list(momentum.columns)[-1] #前一次更新后的最后一天，用于确定下次更新开始的时间
            if last_update_day>=enddate:
                pass
            else:
                startdate=next_trade_date(last_update_day)
                startindex=timelist.index(startdate) #place of startdate in timelist
                endindex=timelist.index(enddate) #place of enddate in timelist
                return_timelist=timelist[startindex:endindex+1]
                get_momentum()
        except FileNotFoundError:
            if is_trade_date(startdate)==0:          
                startdate=next_trade_date(startdate)
            else:
                startdate=startdate
            startindex=timelist.index(startdate) #place of startdate in timelist
            endindex=timelist.index(enddate) #place of enddate in timelist
            return_timelist=timelist[startindex:endindex+1]
            get_momentum()

        try:
            leverage=pd.read_csv(dir_rfhist_path+ '/leverage.csv',error_bad_lines=False).set_index('S_INFO_WINDCODE')
            last_update_day=list(leverage.columns)[-1] #前一次更新后的最后一天，用于确定下次更新开始的时间
            if last_update_day>=enddate:
                pass
            else:
                startdate=next_trade_date(last_update_day)
                startindex=timelist.index(startdate) #place of startdate in timelist
                endindex=timelist.index(enddate) #place of enddate in timelist
                return_timelist=timelist[startindex:endindex+1]
                size=pd.read_csv(dir_rfhist_path+ '/size.csv',error_bad_lines=False).set_index('S_INFO_WINDCODE')
                ME=size[return_timelist]
                get_leverage()
        except FileNotFoundError:
            if is_trade_date(startdate)==0:          
                startdate=next_trade_date(startdate)
            else:
                startdate=startdate
            startindex=timelist.index(startdate) #place of startdate in timelist
            endindex=timelist.index(enddate) #place of enddate in timelist
            return_timelist=timelist[startindex:endindex+1]
            size=pd.read_csv(dir_rfhist_path+ '/size.csv',error_bad_lines=False).set_index('S_INFO_WINDCODE')
            ME=size[return_timelist]
            get_leverage()       

        try:
            liquidity=pd.read_csv(dir_rfhist_path+ '/liquidity.csv',error_bad_lines=False).set_index('S_INFO_WINDCODE')
            last_update_day=list(liquidity.columns)[-1] #前一次更新后的最后一天，用于确定下次更新开始的时间
            if last_update_day>=enddate:
                pass
            else:
                startdate=next_trade_date(last_update_day)
                startindex=timelist.index(startdate) #place of startdate in timelist
                endindex=timelist.index(enddate) #place of enddate in timelist
                return_timelist=timelist[startindex:endindex+1]
                get_liquidity()
        except FileNotFoundError:
            if is_trade_date(startdate)==0:          
                startdate=next_trade_date(startdate)
            else:
                startdate=startdate
            startindex=timelist.index(startdate) #place of startdate in timelist
            endindex=timelist.index(enddate) #place of enddate in timelist
            return_timelist=timelist[startindex:endindex+1]
            get_liquidity()

        all_timelist=get_alltimelist(str(timelist[0]),str(timelist[-1])).suanriqi()   
        try:
            growth=pd.read_csv(dir_rfhist_path+ '/growth.csv',error_bad_lines=False).set_index('S_INFO_WINDCODE')
            last_update_day=list(liquidity.columns)[-1] #前一次更新后的最后一天，用于确定下次更新开始的时间
            if last_update_day>=enddate:
                pass
            else:
                startdate=next_trade_date(last_update_day)
                startindex=timelist.index(startdate) #place of startdate in timelist
                endindex=timelist.index(enddate) #place of enddate in timelist
                return_timelist=timelist[startindex:endindex+1]
                get_SGRO3()
        except FileNotFoundError:
            if is_trade_date(startdate)==0:          
                startdate=next_trade_date(startdate)
            else:
                startdate=startdate
            startindex=timelist.index(startdate) #place of startdate in timelist
            endindex=timelist.index(enddate) #place of enddate in timelist
            return_timelist=timelist[startindex:endindex+1]
            get_SGRO3()

        try:
            earning_yield=pd.read_csv(dir_rfhist_path+ '/earning_yield.csv',error_bad_lines=False).set_index('S_INFO_WINDCODE')
            last_update_day=list(liquidity.columns)[-1] #前一次更新后的最后一天，用于确定下次更新开始的时间
            if last_update_day>=enddate:
                pass
            else:
                startdate=next_trade_date(last_update_day)
                startindex=timelist.index(startdate) #place of startdate in timelist
                endindex=timelist.index(enddate) #place of enddate in timelist
                return_timelist=timelist[startindex:endindex+1]
                get_earning_yield()
        except FileNotFoundError:
            if is_trade_date(startdate)==0:          
                startdate=next_trade_date(startdate)
            else:
                startdate=startdate
            startindex=timelist.index(startdate) #place of startdate in timelist
            endindex=timelist.index(enddate) #place of enddate in timelist
            return_timelist=timelist[startindex:endindex+1]
            get_earning_yield()
        #下面代码更新beta和volatility    
        try:                        
            beta=pd.read_csv(dir_rfhist_path+ '/beta.csv',error_bad_lines=False).set_index('S_INFO_WINDCODE')   
            last_update_day=list(beta.columns)[-1] #前一次更新后的最后一天，用于确定下次更新开始的时间
            if last_update_day>=enddate:
                pass
            else:
                startdate=next_trade_date(last_update_day)
                startindex=timelist.index(startdate) #place of startdate in timelist
                endindex=timelist.index(enddate) #place of enddate in timelist
                return_timelist=timelist[startindex:endindex+1]
                HSIGMA=get_beta_hsigma()
                get_volatility()

        except FileNotFoundError:
            if is_trade_date(startdate)==0:          
                startdate=next_trade_date(startdate)
            else:
                startdate=startdate
            startindex=timelist.index(startdate) #place of startdate in timelist
            endindex=timelist.index(enddate) #place of enddate in timelist
            return_timelist=timelist[startindex:endindex+1]
            HSIGMA=get_beta_hsigma()
            get_volatility()
        #startindex=timelist.index(startdate) #place of startdate in timelist
        #return_timelist=timelist[startindex:endindex+1]
        #print('start')
        #print('beta done')
        '''
        try:
            volatility=pd.read_csv(dir_rfhist_path+ '/volatility.csv',error_bad_lines=False).set_index('S_INFO_WINDCODE')
            last_update_day=list(beta.columns)[-1] #前一次更新后的最后一天，用于确定下次更新开始的时间
            if last_update_day==enddate:
                pass
            elif last_update_day<enddate:
                startdate=next_trade_date(last_update_day)
                startindex=timelist.index(startdate) #place of startdate in timelist
                endindex=timelist.index(enddate) #place of enddate in timelist
                return_timelist=timelist[startindex:endindex+1]
                HSIGMA=get_beta_hsigma()
                get_volatility()
        except FileNotFoundError:
            if is_trade_date(startdate)==0:          
                startdate=next_trade_date(startdate)
            else:
                startdate=startdate
            startindex=timelist.index(startdate) #place of startdate in timelist
            endindex=timelist.index(enddate) #place of enddate in timelist
            return_timelist=timelist[startindex:endindex+1]
            HSIGMA=get_beta_hsigma()
            get_volatility()

        #print('vo done')
        '''


    #end=time.time()
    #print((end-start)/60.0)
