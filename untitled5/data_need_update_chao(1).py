# coding: utf-8

import cx_Oracle
import pandas as pd
import datetime
from datetime import timedelta
import time
import os
import numpy as np
import tushare as ts

#start=datetime.datetime.now()
#start=time.time()
'''
#存储数据路径为：
dir_path=input('Enter dir_path:')
des_path=input('Enter des_path:')
'''
#输入开始/结束日期:
#startdate=input('Enter startdate:')
#enddate=input('Enter enddate:')

dir_path='/usr/datashare/fun_pq_factors'
des_path='/usr/datashare/fun_pq_factors'
#des_path='/usr/datashare/test_chao/fun'
'''
dir_path='/usr/datashare/test_chao/fun_result'
des_path='/usr/datashare/test_chao/fun_result'
'''
'''
dir_path='/usr/datashare/test_chao/fun'
des_path='/usr/datashare/test_chao/fun_result'
'''
#dir_path='/Users/chaowang/Desktop/alpha1'
#des_path='/Users/chaowang/Desktop/alpha1'
#startdate='20180901'
#enddate='20181101'

"""
#若每天跑：
#有问题
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
"""
'''
def get_time():
    # 判断今日是不是周一
    to_weekday = datetime.date.today().weekday()
    # 获取今天的日期
    today = datetime.date.today()
    today_format = today.strftime('%Y%m%d')
    # 周一，获取上周五
    all_timelist_from_year_1990=ts.trade_cal().set_index('calendarDate',inplace=True)
    yesterday = datetime.date.today() + datetime.timedelta(-1)
    yesterday = yesterday.strftime('%Y%m%d')
'''
#下面的三个函数判断某天是不是交易日以及取输入日期的上一个交易日、下一个交易日
def is_trade_date(day):
    trade_date_list = ts.trade_cal().set_index("calendarDate")
    day = pd.to_datetime(day).date()
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
'''
#如果输入的开始时间等于结束时间，就假定更新昨天一个交易日的数据，否则就更新一段时间的数据
if startdate==enddate:
    startdate=last_trade_date(startdate)
    enddate=startdate
'''

def HistData(filename):             #处理历史数据,当不存在历史数据时，让它变成空的dataframe
    try:
        hist_data=pd.read_csv(dir_path+'/'+filename,encoding='gbk',low_memory=False).set_index('S_INFO_WINDCODE')
    except FileNotFoundError:
        hist_data=pd.DataFrame()
    return hist_data

#价量:高开低收均复权价

def getstock_adjfactor(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,trade_dt,s_dq_adjfactor from wind.AShareEODPrices where trade_dt>=%s AND trade_dt<=%s " % (
    x_str, y_str)
    df = pd.read_sql(sql, db)
    d = pd.DataFrame(df)
    d = d.set_index(['TRADE_DT', 'S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    return d

def getstock_avgprice(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    #sql = "select s_info_windcode,trade_dt,S_dq_avgprice from wind.AShareEODPrices where trade_dt>=%s AND trade_dt<=%s order by trade_dt" % (
   # x_str, y_str)
    sql = "select s_info_windcode,trade_dt,s_dq_avgprice from wind.AShareEODPrices where trade_dt>=%s AND trade_dt<=%s order by trade_dt" % (
        x_str, y_str)
    d = pd.read_sql(sql,db)
    d = d.set_index(['TRADE_DT','S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level = 0)
    d.index = d.index.droplevel(1)
    db.close()
    return d

def getstock_avgprice_adj(x,y):
    avgprice_adj = getstock_adjfactor(x,y) * getstock_avgprice(x,y)
    d_hist=HistData('vwap.csv')
    avgprice_adj=pd.concat([d_hist,avgprice_adj],axis=1,sort=True)
    avgprice_adj.index.name='S_INFO_WINDCODE'
    avgprice_adj.to_csv(des_path + '/vwap.csv')
    return 'avgprice_adj done'

def getstock_amount(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,trade_dt,s_dq_amount from wind.AShareEODPrices where trade_dt>=%s AND trade_dt<=%s " %(x_str,y_str)
    df = pd.read_sql(sql, db)
    d = pd.DataFrame(df)
    d = d.set_index(['TRADE_DT', 'S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    d_hist = HistData('amount.csv')
    d = pd.concat([d_hist,d],axis=1,sort=True)
    d.index.name='S_INFO_WINDCODE'
    d.to_csv(des_path+'/amount.csv')
    return 'amount done'


def getstock_volume(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,trade_dt,s_dq_volume from wind.AShareEODPrices where trade_dt>=%s AND trade_dt<=%s " % (
    x_str, y_str)
    df = pd.read_sql(sql, db)
    d = pd.DataFrame(df)
    d = d.set_index(['TRADE_DT', 'S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    d_hist = HistData('volume.csv')
    d = pd.concat([d_hist, d], axis=1,sort=True)
    d.index.name='S_INFO_WINDCODE'
    d.to_csv(des_path+'/volume.csv')
    return 'volume done'


def getstock_openprice_adj(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,trade_dt,s_dq_adjopen from wind.AShareEODPrices where trade_dt>=%s AND trade_dt<=%s " % (
        x_str, y_str)
    df = pd.read_sql(sql, db)
    d = pd.DataFrame(df)
    d = d.set_index(['TRADE_DT', 'S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    d_hist = HistData('openprice_adj.csv')
    d = pd.concat([d_hist, d], axis=1,sort=True)
    d.index.name='S_INFO_WINDCODE'
    d.to_csv(des_path + '/openprice_adj.csv')
    return 'openprice_adj done'

def getstock_highprice_adj(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,trade_dt,s_dq_adjhigh from wind.AShareEODPrices where trade_dt>=%s AND trade_dt<=%s " % (
    x_str, y_str)
    df = pd.read_sql(sql, db)
    d = pd.DataFrame(df)
    d = d.set_index(['TRADE_DT', 'S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    '''
    try:
        d_hist = pd.read_csv(dir_path + '/highprice_adj.csv').set_index('S_INFO_WINDCODE')
    except:
        d_hist = pd.read_csv(dir_path + '/highprice_adj.csv').set_index('Unnamed:0')
    '''
    d_hist=HistData('highprice_adj.csv')
    d = pd.concat([d_hist, d], axis=1,sort=True)
    d.index.name = 'S_INFO_WINDCODE'
    d.to_csv(des_path + '/highprice_adj.csv')
    return 'highprice_adj done'

def getstock_lowprice_adj(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,trade_dt,s_dq_adjlow from wind.AShareEODPrices where trade_dt>=%s AND trade_dt<=%s " % (
    x_str, y_str)
    df = pd.read_sql(sql, db)
    d = pd.DataFrame(df)
    d = d.set_index(['TRADE_DT', 'S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    d_hist = HistData('lowprice_adj.csv')
    d = pd.concat([d_hist, d], axis=1,sort=True)
    d.index.name='S_INFO_WINDCODE'
    d.to_csv(des_path + '/lowprice_adj.csv')
    return 'lowprice_adj done'
'''
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
    d_hist = pd.read_csv(dir_path + '/size.csv').set_index('S_INFO_WINDCODE')
    d1 = pd.concat([d_hist, d], axis=1, sort=True)
    d1.index.name = 'S_INFO_WINDCODE'
    d1.to_csv(des_path + '/size.csv')
    return
getStyle_size(startdate,enddate)
print('size update done')
'''
"""
def getstock_closeprice_adj_and_return(x,y):
    x_str = str(x)
    y_str = str(y)
    #因为计算给定时间段收益率的时候，第一个给定天的收益率全为NaN，所以计算收益率时候需要把给定的时间往前推一天
    x_str=(pd.to_datetime(x_str).date()+datetime.timedelta(-1)).strftime('%Y%m%d')
    #y_str=(pd.to_datetime(y_str).date()+datetime.timedelta(-1)).strftime('%Y%m%d')
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,trade_dt,s_dq_adjclose from wind.AShareEODPrices where trade_dt>=%s AND trade_dt<=%s " % (
    x_str, y_str)
    df = pd.read_sql(sql, db)
    d = pd.DataFrame(df)
    d = d.set_index(['TRADE_DT', 'S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    stock_return=d.diff(axis=1)/d.shift(periods=1,axis=1)
    stock_return.drop(x_str, axis=1, inplace=True)#把第一列全是NaN的收益率删掉，使其变成给定时间段的收益率
    stock_return_hist=pd.read_csv(dir_path+'/Return.csv').set_index('S_INFO_WINDCODE')
    stock_return=pd.concat([stock_return_hist,stock_return],axis=1,sort=True)
    stock_return.index.name = 'S_INFO_WINDCODE'
    stock_return.to_csv(des_path + '/Return.csv')
    d.drop(x_str, axis=1, inplace=True)
    d_hist = pd.read_csv(dir_path + '/closeprice_adj.csv').set_index('S_INFO_WINDCODE')
    d1 = pd.concat([d_hist, d], axis=1,sort=True)
    d1.index.name='S_INFO_WINDCODE'
    d1.to_csv(des_path + '/closeprice_adj.csv')
    return stock_return
    #return d1
stock_return=getstock_closeprice_adj_and_return(startdate,enddate)
print('close and return done')
"""

def getstock_closeprice_adj(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,trade_dt,s_dq_adjclose from wind.AShareEODPrices where trade_dt>=%s AND trade_dt<=%s " % (
    x_str, y_str)
    df = pd.read_sql(sql, db)
    d = pd.DataFrame(df)
    d = d.set_index(['TRADE_DT', 'S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    d_hist = HistData('closeprice_adj.csv')
    d1 = pd.concat([d_hist, d], axis=1,sort=True)
    d1.index.name='S_INFO_WINDCODE'
    d1.to_csv(des_path + '/closeprice_adj.csv')
    return d1
#他、return需要从14年开始
#收益率：
def getstock_return(startdate,enddate):
    stock_return=close_adj.diff(axis=1)/close_adj.shift(periods=1,axis=1)
    stock_return=stock_return.loc[:,startdate:enddate]
    flag_1=flag.loc[:,startdate:enddate]
    stock_return_new=stock_return*flag_1
    d_hist = HistData('Return.csv')
    d1 = pd.concat([d_hist,stock_return_new],axis=1,sort=True)
    d1.index.name='S_INFO_WINDCODE'
    d1.to_csv(des_path + '/Return.csv')
    return 'Return done'
#circulation_value as weight
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
    d_hist = HistData('weight.csv')
    d1=pd.concat([d_hist,d],axis=1,sort=True)
    d1.index.name='S_INFO_WINDCODE'
    d1.to_csv(des_path+'/weight.csv')
    return 'circulation_value done'

def get_industry_new():
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl",encoding='gbk') #不用gbk读不出来行业
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
    db.close()
    dummies = pd.get_dummies(industry, columns=['SECTOR'], prefix=['Industry'], prefix_sep="_", dummy_na=False,drop_first=False)
    temp=pd.DataFrame(index = stock_list,columns = dummies.columns)
    temp.loc[:,:]=dummies.loc[:,:]
    temp.index.name='S_INFO_WINDCODE'
    return temp




#获取所有日期/all_timelist/用做fill:
class get_alltimelist():   
    """
    取区间日期模块，主要传入两个参数:
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

#benchmark类
def geths300_return(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,trade_dt,pct_chg from wind.HS300IEODPrices where trade_dt>=%s AND trade_dt<=%s order by trade_dt" % (
    x_str, y_str)
    d = pd.read_sql(sql,db)
    d = d.set_index(['TRADE_DT', 'S_INFO_WINDCODE'])
    d = d.stack(level = 0)
    d = d.unstack(level = 0)
    d.index = d.index.droplevel(1)
    d_hist =HistData('hs300_return.csv')
    d = pd.concat([d_hist, d], axis=1,sort=True)
    d.index.name='S_INFO_WINDCODE'
    d.to_csv(des_path + '/hs300_return.csv')
    db.close()
    return 'hs300_return done'


def geths300_price(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,trade_dt,s_dq_close from wind.HS300IEODPrices where trade_dt>=%s AND trade_dt<=%s order by trade_dt" % (
    x_str, y_str)
    d = pd.read_sql(sql,db)
    d = d.set_index(['TRADE_DT', 'S_INFO_WINDCODE'])
    d = d.stack(level = 0)
    d = d.unstack(level = 0)
    d.index = d.index.droplevel(1)
    db.close()
    d_hist = HistData('hs300_price.csv')
    d = pd.concat([d_hist, d], axis=1,sort=True)
    d.index.name='S_INFO_WINDCODE'
    d.to_csv(des_path + '/hs300_price.csv')
    return 'hs300_price done'



#统一处理基本面因子的方式：
#使用之前要set index，读取老数据
def fmtfactor_adj(df_hist,df):
    if df.empty!=True:
        #非空，处理过后fill；
        df=df.stack(level=0)
        df=df.unstack(level=0)
        df.index=df.index.droplevel(1)
        #先生成这一段时间的
        df_fill1=pd.DataFrame(index=stock_list,columns=all_timelist) #this period of time
        df_fill1.loc[:,:]=df.loc[:,:]
    else:
        #empty,generate new and then fill
        df_fill1 = pd.DataFrame(index=stock_list, columns=all_timelist) #this period of time
    df_fill1.index.name = 'S_INFO_WINDCODE'
    df_fill3=pd.DataFrame(index=stock_list,columns=return_timelist)
    df_fill3.loc[:,:]=df_fill1.loc[:,:] 
    #与老表合并再向前fill
    df_fill2 = pd.concat([df_hist,df_fill3],axis=1,sort=True) #hist+present
    df_fill2=df_fill2.fillna(method='pad',axis=1)
    #df_fill3=pd.DataFrame(index=stock_list,columns=timelist)
    #df_fill3.loc[:,:]=df_fill2.loc[:,:] #hist+present
    return df_fill2

#risk-factor类:
def get_longterm_liability(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,report_period,tot_non_cur_liab from wind.AShareBalanceSheet where statement_type=408001000 AND report_period>=%s AND report_period<=%s order by report_period " % (
    x_str, y_str)
    df = pd.read_sql(sql, db)
    d = pd.DataFrame(df)
    d_hist = HistData('Longterm_Liability.csv')
    if d.empty!=True:
        d = d.set_index(['REPORT_PERIOD', 'S_INFO_WINDCODE'])
        d1 = fmtfactor_adj(d_hist,d)
        #d2 = pd.concat([d_hist,d1],axis=1,sort=True)
    else:
        d1 = fmtfactor_adj(d_hist, d)
        #d2 = pd.concat([d_hist, d1], axis=,sort=True1)
    d1.index.name='S_INFO_WINDCODE'
    d1.to_csv(des_path+'/Longterm_Liability.csv')
    db.close()
    return 'longterm_liability done'

def get_TotalDebt(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,report_period,tot_liab from wind.AShareBalanceSheet where statement_type=408001000 AND report_period>=%s AND report_period<=%s order by report_period " % (
    x_str, y_str)
    df = pd.read_sql(sql, db)
    d = pd.DataFrame(df)
    d_hist = HistData('Total_debt.csv')
    if d.empty!=True:
        d = d.set_index(['REPORT_PERIOD', 'S_INFO_WINDCODE'])
        d1 = fmtfactor_adj(d_hist,d)
        #d2 = pd.concat([d_hist,d1],axis=1,sort=True)
    else:
        d1 = fmtfactor_adj(d_hist, d)
        #d2 = pd.concat([d_hist, d1], axis=1,sort=True)
    d1.index.name='S_INFO_WINDCODE'
    d1.to_csv(des_path + '/Total_debt.csv')
    db.close()
    return 'Total_Debt done'

def get_TotalAsset(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,report_period,tot_assets from wind.AShareBalanceSheet where statement_type=408001000 AND report_period>=%s AND report_period<=%s order by report_period " % (
    x_str, y_str)
    df = pd.read_sql(sql, db)
    d = pd.DataFrame(df)
    d_hist = HistData('Total_Asset.csv')
    #d = d.set_index(['REPORT_PERIOD', 'S_INFO_WINDCODE'])
    if d.empty!=True:
        d = d.set_index(['REPORT_PERIOD', 'S_INFO_WINDCODE'])
        d1 = fmtfactor_adj(d_hist,d)
        #d2 = pd.concat([d_hist,d1],axis=1,sort=True)
    else:
        d1 = fmtfactor_adj(d_hist, d)
        #d2 = pd.concat([d_hist, d1], axis=1,sort=True)
    d1.index.name='S_INFO_WINDCODE'
    db.close()
    d1.to_csv(des_path + '/Total_Asset.csv')
    return 'Total_Asset done'

def get_BookEquity(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,report_period,tot_shrhldr_eqy_excl_min_int from wind.AShareBalanceSheet where statement_type=408001000 AND report_period>=%s AND report_period<=%s order by report_period " % (
    x_str, y_str)
    df = pd.read_sql(sql, db)
    d = pd.DataFrame(df)
    d_hist = HistData('Book_Equity.csv')
    if d.empty!=True:
        d = d.set_index(['REPORT_PERIOD', 'S_INFO_WINDCODE'])
        d1 = fmtfactor_adj(d_hist,d)
        #d2 = pd.concat([d_hist,d1],axis=1,sort=True)
    else:
        d1 = fmtfactor_adj(d_hist, d)
        #d2 = pd.concat([d_hist, d1], axis=1,sort=True)
    d1.index.name='S_INFO_WINDCODE'
    db.close()
    d1.to_csv(des_path + '/Book_Equity.csv')
    return 'Book_Equity done'

def get_circulation_capitalStock(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,trade_dt,float_a_shr_today from wind.AShareEODDerivativeIndicator where trade_dt>=%s AND trade_dt<=%s " % (
    x_str, y_str)
    df = pd.read_sql(sql, db)
    d = pd.DataFrame(df)
    d = d.set_index(['TRADE_DT', 'S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    d.index = d.index.droplevel(1)
    CCS_new = pd.DataFrame(index=stock_list, columns=all_timelist)
    CCS_new.loc[:, :] = d.loc[:, :]
    CCS_new2=pd.DataFrame(index=stock_list, columns=return_timelist)
    CCS_new2.loc[:, :] = CCS_new.loc[:, :]
    d_hist = HistData('circulation_capitalStock.csv')
    d1 = pd.concat([d_hist, CCS_new2], axis=1,sort=True)
    d1.index.name='S_INFO_WINDCODE'
    d1.to_csv(des_path + '/circulation_capitalStock.csv')
    db.close()
    return 'circulation_capitalstock done'


def get_operation_revenue(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,report_period,tot_oper_rev from wind.AShareIncome where statement_type=408001000 AND report_period>=%s AND report_period<=%s order by report_period " %(x_str, y_str)
    df = pd.read_sql(sql, db)
    d = pd.DataFrame(df)
    if d.empty!=True:
        d = d.set_index(['REPORT_PERIOD', 'S_INFO_WINDCODE'])
        d = d.stack(level=0)
        d = d.unstack(level=0)
        d.index = d.index.droplevel(1)
        d_hist = HistData('Operation_Revenue.csv')
        d1 = pd.concat([d_hist, d], axis=1,sort=True)
        d1.index.name='S_INFO_WINDCODE'
        d1.to_csv(des_path + '/Operation_Revenue.csv')
    else:
        d_hist = HistData('Operation_Revenue.csv')
        d_hist.to_csv(des_path + '/Operation_Revenue.csv')
    db.close()
    return 'Operation_Revenue done'

#ss is single season
def get_operation_revenue_ss(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,report_period,ann_dt,tot_oper_rev from wind.AShareIncome where statement_type=408002000 AND report_period>=%s AND report_period<=%s order by report_period " %(x_str, y_str)
    df = pd.read_sql(sql, db)
    if df.empty!=True:
        gor = pd.DataFrame(df)
        gor1=gor.sort_values(by=['S_INFO_WINDCODE','REPORT_PERIOD','ANN_DT'])
        gor2=gor1.groupby(['S_INFO_WINDCODE','REPORT_PERIOD']).tail(1)
        gor2 = gor2.set_index(['REPORT_PERIOD','S_INFO_WINDCODE'])
        del(gor2['ANN_DT'])
        gor2 = gor2.stack(level=0)
        gor2 = gor2.unstack(level=0)
        gor2.index = gor2.index.droplevel(1)
        d_hist = HistData('OR(single season).csv')
        d1 = pd.concat([d_hist, gor2], axis=1,sort=True)
        d1.index.name='S_INFO_WINDCODE'
        d1.to_csv(des_path + '/OR(single season).csv')
    else:
        d_hist = HistData('OR(single season).csv')
        d_hist.to_csv(des_path + '/OR(single season).csv')

    db.close()
    return 'OR_ss done'

def get_pe_ttm(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,trade_dt,s_val_pe_ttm from wind.AShareEODDerivativeIndicator where trade_dt>=%s AND trade_dt<=%s order by trade_dt" % (
    x_str, y_str)
    d = pd.read_sql(sql,db)
    d = d.set_index(['TRADE_DT', 'S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    d2=pd.DataFrame(index=stock_list,columns=return_timelist)
    d2.loc[:,:]=d.loc[:,:]
    d_hist = HistData('pe.csv')
    d1 = pd.concat([d_hist, d2], axis=1,sort=True)
    d1.index.name='S_INFO_WINDCODE'
    d1.to_csv(des_path+'/pe.csv')
    return 'pe done'

def get_eps_est(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,est_dt,est_report_dt,eps_avg from wind.AShareConsensusData where est_dt>=%s AND est_dt<=%s order by s_info_windcode" %(x_str, y_str)
    df = pd.read_sql(sql, db)
    d = pd.DataFrame(df)
    d = d.set_index('S_INFO_WINDCODE')
    d_hist = HistData('eps_est.csv')
    if d.empty!=True:#d不为空
        #先处理历史在有数据之后的情况
        d1 = d.groupby(['S_INFO_WINDCODE', 'EST_DT']).mean()
        d2 = d1.unstack(level=0).T
        d2.index = d2.index.droplevel(0)
        d2 = d2.fillna(method='pad', axis=1)
        d3 = pd.DataFrame(index=stock_list, columns=all_timelist)
        d3.loc[:, :] = d2.loc[:, :]
        d3 = d3.fillna(method='pad', axis=1)
        d4 = pd.DataFrame(index=stock_list, columns=return_timelist)
        d4.loc[:, :] = d3.loc[:, :]
        #再合并，处理中间没有数据的部分
        d_new=pd.concat([d_hist,d4],axis=1,sort=True)
        d_new=d_new.fillna(method='pad', axis=1)
    else:#d为空
        #生成空表
        d2=pd.DataFrame(index=stock_list, columns=return_timelist)
        d2.index.name = 'S_INFO_WINDCODE'
        d_new = pd.concat([d_hist, d2], axis=1,sort=True)
        d_new = d_new.fillna(method='pad', axis=1)
    d_new.index.name='S_INFO_WINDCODE'
    db.close()
    d_new.to_csv(des_path+'/eps_est.csv')
    return 'eps_est done'

def getStock_ST():
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,entry_dt,remove_dt,s_type_st from wind.AShareST"
    d = pd.read_sql(sql,db)
    db.close()
    d.sort_values(by = 'S_INFO_WINDCODE',inplace= True)
    return d

def getStock_IPO():
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,s_ipo_listdate from wind.AShareIPO"
    d = pd.read_sql(sql,db)
    db.close()
    d.sort_values(by = 'S_INFO_WINDCODE',inplace = True)
    return d

def getStock_status(x,y):#查看股票的状态，看是否为交易状态还是停牌状态
    x_str=str(x)
    y_str=str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl",encoding='gbk')
    sql = "select s_info_windcode,s_dq_tradestatus,trade_dt from wind.AShareEODPrices where trade_dt>=%s AND trade_dt<=%s " % (
    x_str, y_str)
    d = pd.read_sql(sql,db)
    d = d.set_index(['TRADE_DT','S_INFO_WINDCODE'])
    d =d.stack(level = 0)
    d = d.unstack(level = 0)
    db.close()
    d.index = d.index.droplevel(1)
    d_hist=HistData('stock_status.csv')
    d=pd.concat([d_hist,d],axis=1,sort=True)
    d.index.name='S_INFO_WINDCODE'
    d.to_csv(des_path+'/stock_status.csv',encoding='gbk')  #dataframe中的内容为中文时存储时候会发生乱码，需要加上encoding这句
    return d

def getStockTechData_high_price(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,trade_dt,s_dq_high from wind.AShareEODPrices where trade_dt>=%s AND trade_dt<=%s " % (
    x_str, y_str)
    df = pd.read_sql(sql, db)
    d = pd.DataFrame(df)
    d = d.set_index(['TRADE_DT', 'S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    d_hist=HistData('highprice.csv')
    d=pd.concat([d_hist,d],axis=1,sort=True)
    d.index.name='S_INFO_WINDCODE'
    d.to_csv(des_path+'/highprice.csv')
    return d

def getStockTechData_low_price(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,trade_dt,s_dq_low from wind.AShareEODPrices where trade_dt>=%s AND trade_dt<=%s " % (
    x_str, y_str)
    df = pd.read_sql(sql, db)
    d = pd.DataFrame(df)
    d = d.set_index(['TRADE_DT', 'S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    d_hist=HistData('lowprice.csv')
    d=pd.concat([d_hist,d],axis=1,sort=True)
    d.index.name='S_INFO_WINDCODE'
    d.to_csv(des_path+'/lowprice.csv')
    return d

def get_flag(): #把不可用的数据点的位置设置为NAN，然后乘以return的矩阵，达到剔除的效果（代码可精简）
    #获取一个比较的对象/矩阵
    flag = pd.DataFrame(index = stock_list,columns = timelist)
    flag.iloc[:,:] = 1
    #stock_list=list(close_adj.index) #所有股票
    #date_read_list=list(close_adj.columns) #所有日期
    start_dt=timelist[0] #有可能开始/结束读取数据的日期不是交易日，更精确一步获取开始与结束的日子
    end_dt=timelist[-1]

    #处理ST
    for code in stock_list:
        ST=st_state[st_state.S_INFO_WINDCODE==code]
        if ST.empty==True:
            continue
        else:
            ST=ST.fillna('20200101')
            #获取每只股票每次特殊处理的开始/结束时间
        for i in range(0,len(ST)):
                entry_dt=int(ST.iloc[i,1])
                remove_dt=int(ST.iloc[i,2])

                #对ST处理的标注时间进行处理
                #ST时间在读取数据期间：
                if int(start_dt)<=entry_dt<=int(end_dt) and int(start_dt)<=remove_dt<=int(end_dt):
                    ST_begin_dt=entry_dt
                    ST_end_dt=remove_dt

                #ST结束时间落在读取数据期间：
                elif entry_dt<int(start_dt) and int(start_dt)<=remove_dt<=int(end_dt):
                    ST_begin_dt=start_dt
                    ST_end_dt=remove_dt

                #ST开始时间落在读取数据期间：
                elif int(start_dt)<=entry_dt<=int(end_dt) and remove_dt>=int(end_dt):
                    ST_begin_dt=entry_dt
                    ST_end_dt=end_dt

                #ST区间包括了读数据的区间：
                elif entry_dt<int(start_dt) and remove_dt>=int(end_dt):
                    ST_begin_dt=entry_dt
                    ST_end_dt=remove_dt

                #其他情况不需要考虑
                else:
                    continue

                flag.loc[code,str(ST_begin_dt):str(ST_end_dt)] = None

    #处理IPO
    for code in stock_list:
        ipo1 = ipo_Date[ipo_Date.S_INFO_WINDCODE == code]
        #获取IPO发行的起始时间,有些退市的股票例如000022.SZ就不在ipo的那张表里，所以需要下面的continue语句
        if ipo1.empty:
            continue
        else:
            ipo_start_dt = (ipo1.iloc[0,1])
            ipo_remove_dt=(pd.to_datetime(ipo_start_dt)+timedelta(365)).strftime('%Y%m%d')
            #对IPO处理的标注时间进行处理
            #IPO时间在读取数据期间：
            if int(start_dt)<=int(ipo_start_dt)<=int(end_dt) and int(start_dt)<=int(ipo_remove_dt)<=int(end_dt):
                ipo_begin_dt=ipo_start_dt
                ipo_end_dt=ipo_remove_dt
            #IPO结束时间落在读取数据期间：
            elif int(ipo_start_dt)<int(start_dt) and int(start_dt)<=int(ipo_remove_dt)<=int(end_dt):
                ipo_begin_dt=start_dt
                ipo_end_dt=ipo_remove_dt
            #IPO开始时间落在读取数据期间：
            elif int(start_dt)<=int(ipo_start_dt)<=int(end_dt) and int(ipo_remove_dt)>=int(end_dt):
                ipo_begin_dt=ipo_start_dt
                ipo_end_dt=end_dt
            #IPO时间段包括了读取数据的时间段：
            elif int(ipo_start_dt)<int(start_dt) and int(ipo_remove_dt)>int(end_dt):
                ipo_begin_dt=start_dt
                ipo_end_dt=end_dt
            #其他情况不需要考虑
            else:
                continue
            flag.loc[code,str(ipo_begin_dt):str(ipo_end_dt)] = None
    #下面处理停牌和涨跌停
    temp=stock_status=='停牌'
    flag[temp]=np.nan
    temp_11=highprice==lowprice
    temp_12=stock_return_temp> 0.099     #用于剔除涨停
    temp_13=stock_return_temp< -0.099    #用于剔除跌停
    temp_14=temp_12 +temp_13
    temp_15=temp_11*temp_14
    flag[temp_15]=np.nan
    #下面标记退市
    temp=stock_status.fillna(0)
    flag[temp==0]=np.nan
    flag.index.name='S_INFO_WINDCODE'
    flag.to_csv(des_path+'/flag.csv')
    return flag

#startdate='20180602'
#enddate='20181105'
#当dir_path重中历史数据不存在时候，只需要更改下面的startdate和enddate,就能读取给定时间段的基本面数据
startdate=datetime.date.today().strftime('%Y%m%d')
enddate=datetime.date.today().strftime('%Y%m%d')
if is_trade_date(enddate)==0:
    pass
else:
    enddate=last_trade_date(enddate)
    #下面每个try...except语句用于检查每个基本面因子的最近的更新日期
    try:
        vwap=pd.read_csv(dir_path+'/vwap.csv') #用于更新检查日期的指标
        last_update_day=list(vwap.columns)[-1]  #前一次更新后的最后一天，用于确定下次更新开始的时间
        if last_update_day>=enddate:
            pass
        else:
            startdate=next_trade_date(last_update_day)
            getstock_avgprice_adj(startdate,enddate)        
    except FileNotFoundError:
        if is_trade_date(startdate)==0:          
            startdate=next_trade_date(startdate)
        else:
            startdate=startdate
        getstock_avgprice_adj(startdate,enddate)
    #print(enddate)
    #print(startdate)   
    
    try:
        amount=pd.read_csv(dir_path+'/amount.csv') #用于更新检查日期的指标
        last_update_day=list(amount.columns)[-1]  #前一次更新后的最后一天，用于确定下次更新开始的时间
        if last_update_day>=enddate:
            pass
        else:
            startdate=next_trade_date(last_update_day)
            getstock_amount(startdate,enddate)        
    except FileNotFoundError:
        if is_trade_date(startdate)==0:          
            startdate=next_trade_date(startdate)
        else:
            startdate=startdate
        getstock_amount(startdate,enddate)
    
    try:
        volume=pd.read_csv(dir_path+'/volume.csv') #用于更新检查日期的指标
        last_update_day=list(volume.columns)[-1]  #前一次更新后的最后一天，用于确定下次更新开始的时间
        if last_update_day>=enddate:
            pass
        else:
            startdate=next_trade_date(last_update_day)
            getstock_volume(startdate,enddate)        
    except FileNotFoundError:
        if is_trade_date(startdate)==0:          
            startdate=next_trade_date(startdate)
        else:
            startdate=startdate
        getstock_volume(startdate,enddate)
    
    try:
        openprice_adj=pd.read_csv(dir_path+'/openprice_adj.csv') #用于更新检查日期的指标
        last_update_day=list(openprice_adj.columns)[-1]  #前一次更新后的最后一天，用于确定下次更新开始的时间
        if last_update_day>=enddate:
            pass
        else:
            startdate=next_trade_date(last_update_day)
            getstock_openprice_adj(startdate,enddate)       
    except FileNotFoundError:
        if is_trade_date(startdate)==0:          
            startdate=next_trade_date(startdate)
        else:
            startdate=startdate
        getstock_openprice_adj(startdate,enddate)
    
    try:
        highprice_adj=pd.read_csv(dir_path+'/highprice_adj.csv') #用于更新检查日期的指标
        last_update_day=list(highprice_adj.columns)[-1]  #前一次更新后的最后一天，用于确定下次更新开始的时间
        if last_update_day>=enddate:
            pass
        else:
            startdate=next_trade_date(last_update_day)
            getstock_highprice_adj(startdate,enddate)
    except FileNotFoundError:
        if is_trade_date(startdate)==0:          
            startdate=next_trade_date(startdate)
        else:
            startdate=startdate
        getstock_highprice_adj(startdate,enddate)
    
    try:
        lowprice_adj=pd.read_csv(dir_path+'/lowprice_adj.csv') #用于更新检查日期的指标
        last_update_day=list(lowprice_adj.columns)[-1]  #前一次更新后的最后一天，用于确定下次更新开始的时间
        if last_update_day>=enddate:
            pass
        else:
            startdate=next_trade_date(last_update_day)
            getstock_lowprice_adj(startdate,enddate)
    except FileNotFoundError:
        if is_trade_date(startdate)==0:          
            startdate=next_trade_date(startdate)
        else:
            startdate=startdate
        getstock_lowprice_adj(startdate,enddate)
    
    try:
        closeprice_adj=pd.read_csv(dir_path+'/closeprice_adj.csv') #用于更新检查日期的指标
        last_update_day=list(closeprice_adj.columns)[-1]  #前一次更新后的最后一天，用于确定下次更新开始的时间
        if last_update_day>=enddate:
            pass
        else:
            startdate=next_trade_date(last_update_day)
            getstock_closeprice_adj(startdate,enddate)
    except FileNotFoundError:
        if is_trade_date(startdate)==0:          
            startdate=next_trade_date(startdate)
        else:
            startdate=startdate
        getstock_closeprice_adj(startdate,enddate)
    #获取一些对准的指标：
    #获取交易日期/timelist/用做column index:
    close_adj=pd.read_csv(dir_path+'/closeprice_adj.csv').set_index('S_INFO_WINDCODE')
    timelist=list(close_adj.columns)
    #获取所有股票/stock_list/用做row index:
    stock_list=list(close_adj.index)
    st_state = getStock_ST()
    ipo_Date = getStock_IPO()
    try:
        stock_status=pd.read_csv(dir_path+'/stock_status.csv',encoding='gbk',low_memory=False) #用于更新检查日期的指标
        last_update_day=list(stock_status.columns)[-1]  #前一次更新后的最后一天，用于确定下次更新开始的时间
        if last_update_day>=enddate:
            pass
        else:
            startdate=next_trade_date(last_update_day)
            getStock_status(startdate,enddate)       
    except FileNotFoundError:
        if is_trade_date(startdate)==0:          
            startdate=next_trade_date(startdate)
        else:
            startdate=startdate
        getStock_status(startdate,enddate)
    stock_status=pd.read_csv(dir_path+'/stock_status.csv',encoding='gbk',low_memory=False).set_index('S_INFO_WINDCODE')
    
    try:
        highprice=pd.read_csv(dir_path+'/highprice.csv') #用于更新检查日期的指标
        last_update_day=list(highprice.columns)[-1]  #前一次更新后的最后一天，用于确定下次更新开始的时间
        if last_update_day>=enddate:
            pass
        else:
            startdate=next_trade_date(last_update_day)
            getStockTechData_high_price(startdate,enddate)       
    except FileNotFoundError:
        if is_trade_date(startdate)==0:          
            startdate=next_trade_date(startdate)
        else:
            startdate=startdate
        getStockTechData_high_price(startdate,enddate)
    highprice=pd.read_csv(dir_path+'/highprice.csv').set_index('S_INFO_WINDCODE')
    
    try:
        lowprice=pd.read_csv(dir_path+'/lowprice.csv') #用于更新检查日期的指标
        last_update_day=list(lowprice.columns)[-1]  #前一次更新后的最后一天，用于确定下次更新开始的时间
        if last_update_day>=enddate:
            pass
        else:
            startdate=next_trade_date(last_update_day)
            getStockTechData_low_price(startdate,enddate)       
    except FileNotFoundError:
        if is_trade_date(startdate)==0:          
            startdate=next_trade_date(startdate)
        else:
            startdate=startdate
        getStockTechData_low_price(startdate,enddate)
    lowprice=pd.read_csv(dir_path+'/lowprice.csv').set_index('S_INFO_WINDCODE')
    stock_return_temp=close_adj.diff(axis=1)/close_adj.shift(periods=1,axis=1)
    print('flag begin')
    flag=get_flag()
    
    try:
        Return=pd.read_csv(dir_path+'/Return.csv') #用于更新检查日期的指标
        last_update_day=list(Return.columns)[-1]  #前一次更新后的最后一天，用于确定下次更新开始的时间
        if last_update_day>=enddate:
            pass
        else:
            startdate=next_trade_date(last_update_day)
            getstock_return(startdate,enddate)       
    except FileNotFoundError:
        if is_trade_date(startdate)==0:          
            startdate=next_trade_date(startdate)
        else:
            startdate=startdate
        getstock_return(startdate,enddate)
    
    try:
        weight=pd.read_csv(dir_path+'/weight.csv') #用于更新检查日期的指标
        last_update_day=list(weight.columns)[-1]  #前一次更新后的最后一天，用于确定下次更新开始的时间
        if last_update_day>=enddate:
            pass
        else:
            startdate=next_trade_date(last_update_day)
            getStock_Circulation_value(startdate,enddate)       
    except FileNotFoundError:
        if is_trade_date(startdate)==0:          
            startdate=next_trade_date(startdate)
        else:
            startdate=startdate
        getStock_Circulation_value(startdate,enddate)
    industry = get_industry_new()
    industry.to_csv(des_path+'/industry.csv')
    
    del close_adj,flag
    try:
        hs300_return=pd.read_csv(dir_path+'/hs300_return.csv') #用于更新检查日期的指标
        last_update_day=list(hs300_return.columns)[-1]  #前一次更新后的最后一天，用于确定下次更新开始的时间
        if last_update_day>=enddate:
            pass
        else:
            startdate=next_trade_date(last_update_day)
            geths300_return(startdate,enddate)       
    except FileNotFoundError:
        if is_trade_date(startdate)==0:          
            startdate=next_trade_date(startdate)
        else:
            startdate=startdate
        geths300_return(startdate,enddate)
    
    try:
        hs300_price=pd.read_csv(dir_path+'/hs300_price.csv') #用于更新检查日期的指标
        last_update_day=list(hs300_price.columns)[-1]  #前一次更新后的最后一天，用于确定下次更新开始的时间
        if last_update_day>=enddate:
            pass
        else:
            startdate=next_trade_date(last_update_day)
            geths300_price(startdate,enddate)       
    except FileNotFoundError:
        if is_trade_date(startdate)==0:          
            startdate=next_trade_date(startdate)
        else:
            startdate=startdate
        geths300_price(startdate,enddate)
    
    try:
        Longterm_Liability=pd.read_csv(dir_path+'/Longterm_Liability.csv') #用于更新检查日期的指标
        last_update_day=list(Longterm_Liability.columns)[-1]  #前一次更新后的最后一天，用于确定下次更新开始的时间
        if last_update_day>=enddate:
            pass
        else:
            startdate=next_trade_date(last_update_day)
            all_timelist=get_alltimelist(str(startdate),str(enddate)).suanriqi()
            start_index=timelist.index(startdate)
            end_index=timelist.index(enddate)
            return_timelist=timelist[start_index:end_index+1]#only需要更新的时间段
            get_longterm_liability(startdate,enddate)       
    except FileNotFoundError:
        if is_trade_date(startdate)==0:          
            startdate=next_trade_date(startdate)
        else:
            startdate=startdate
        all_timelist=get_alltimelist(str(startdate),str(enddate)).suanriqi()
        start_index=timelist.index(startdate)
        end_index=timelist.index(enddate)
        return_timelist=timelist[start_index:end_index+1]#only需要更新的时间段
        get_longterm_liability(startdate,enddate)
    try:
        Total_debt=pd.read_csv(dir_path+'/Total_debt.csv') #用于更新检查日期的指标
        last_update_day=list(Total_debt.columns)[-1]  #前一次更新后的最后一天，用于确定下次更新开始的时间
        if last_update_day>=enddate:
            pass
        else:
            startdate=next_trade_date(last_update_day)
            all_timelist=get_alltimelist(str(startdate),str(enddate)).suanriqi()
            start_index=timelist.index(startdate)
            end_index=timelist.index(enddate)
            return_timelist=timelist[start_index:end_index+1]
            get_TotalDebt(startdate,enddate)       
    except FileNotFoundError:
        if is_trade_date(startdate)==0:          
            startdate=next_trade_date(startdate)
        else:
            startdate=startdate
        all_timelist=get_alltimelist(str(startdate),str(enddate)).suanriqi()
        start_index=timelist.index(startdate)
        end_index=timelist.index(enddate)
        return_timelist=timelist[start_index:end_index+1]#only需要更新的时间段
        get_TotalDebt(startdate,enddate)
    
    try:
        Total_Asset=pd.read_csv(dir_path+'/Total_Asset.csv') #用于更新检查日期的指标
        last_update_day=list(Total_Asset.columns)[-1]  #前一次更新后的最后一天，用于确定下次更新开始的时间
        if last_update_day>=enddate:
            pass
        else:
            startdate=next_trade_date(last_update_day)
            all_timelist=get_alltimelist(str(startdate),str(enddate)).suanriqi()
            start_index=timelist.index(startdate)
            end_index=timelist.index(enddate)
            return_timelist=timelist[start_index:end_index+1]
            get_TotalAsset(startdate,enddate)       
    except FileNotFoundError:
        if is_trade_date(startdate)==0:          
            startdate=next_trade_date(startdate)
        else:
            startdate=startdate
        all_timelist=get_alltimelist(str(startdate),str(enddate)).suanriqi()
        start_index=timelist.index(startdate)
        end_index=timelist.index(enddate)
        return_timelist=timelist[start_index:end_index+1]#only需要更新的时间段
        get_TotalAsset(startdate,enddate)
    
    try:
        Book_Equity=pd.read_csv(dir_path+'/Book_Equity.csv') #用于更新检查日期的指标
        last_update_day=list(Book_Equity.columns)[-1]  #前一次更新后的最后一天，用于确定下次更新开始的时间
        if last_update_day>=enddate:
            pass
        else:
            startdate=next_trade_date(last_update_day)
            all_timelist=get_alltimelist(str(startdate),str(enddate)).suanriqi()
            start_index=timelist.index(startdate)
            end_index=timelist.index(enddate)
            return_timelist=timelist[start_index:end_index+1]
            get_BookEquity(startdate,enddate)      
    except FileNotFoundError:
        if is_trade_date(startdate)==0:          
            startdate=next_trade_date(startdate)
        else:
            startdate=startdate
        all_timelist=get_alltimelist(str(startdate),str(enddate)).suanriqi()
        start_index=timelist.index(startdate)
        end_index=timelist.index(enddate)
        return_timelist=timelist[start_index:end_index+1]#only需要更新的时间段
        get_BookEquity(startdate,enddate)
    try:
        circulation_capitalStock=pd.read_csv(dir_path+'/circulation_capitalStock.csv') #用于更新检查日期的指标
        last_update_day=list(circulation_capitalStock.columns)[-1]  #前一次更新后的最后一天，用于确定下次更新开始的时间
        if last_update_day>=enddate:
            pass
        else:
            startdate=next_trade_date(last_update_day)
            all_timelist=get_alltimelist(str(startdate),str(enddate)).suanriqi()
            start_index=timelist.index(startdate)
            end_index=timelist.index(enddate)
            return_timelist=timelist[start_index:end_index+1]
            get_circulation_capitalStock(startdate,enddate)       
    except FileNotFoundError:
        if is_trade_date(startdate)==0:          
            startdate=next_trade_date(startdate)
        else:
            startdate=startdate
        all_timelist=get_alltimelist(str(startdate),str(enddate)).suanriqi()
        start_index=timelist.index(startdate)
        end_index=timelist.index(enddate)
        return_timelist=timelist[start_index:end_index+1]#only需要更新的时间段
        get_circulation_capitalStock(startdate,enddate)
    
    try:
        Operation_Revenue=pd.read_csv(dir_path+'/Operation_Revenue.csv') #用于更新检查日期的指标
        last_update_day=list(Operation_Revenue.columns)[-1]  #前一次更新后的最后一天，用于确定下次更新开始的时间
        if last_update_day>=enddate:
            pass
        else:
            startdate=next_trade_date(last_update_day)
            get_operation_revenue(startdate,enddate)       
    except FileNotFoundError:
        if is_trade_date(startdate)==0:          
            startdate=next_trade_date(startdate)
        else:
            startdate=startdate
        get_operation_revenue(startdate,enddate)
    try:
        OR=pd.read_csv(dir_path+'/OR(single season).csv') #用于更新检查日期的指标
        last_update_day=list(OR.columns)[-1]  #前一次更新后的最后一天，用于确定下次更新开始的时间
        if last_update_day>=enddate:
            pass
        else:
            startdate=next_trade_date(last_update_day)
            get_operation_revenue_ss(startdate,enddate)       
    except FileNotFoundError:
        if is_trade_date(startdate)==0:          
            startdate=next_trade_date(startdate)
        else:
            startdate=startdate
        get_operation_revenue_ss(startdate,enddate)
    try:
        pe=pd.read_csv(dir_path+'/pe.csv') #用于更新检查日期的指标
        last_update_day=list(pe.columns)[-1]  #前一次更新后的最后一天，用于确定下次更新开始的时间
        if last_update_day>=enddate:
            pass
        else:
            startdate=next_trade_date(last_update_day)
            all_timelist=get_alltimelist(str(startdate),str(enddate)).suanriqi()
            start_index=timelist.index(startdate)
            end_index=timelist.index(enddate)
            return_timelist=timelist[start_index:end_index+1]
            get_pe_ttm(startdate,enddate)       
    except FileNotFoundError:
        if is_trade_date(startdate)==0:          
            startdate=next_trade_date(startdate)
        else:
            startdate=startdate
        all_timelist=get_alltimelist(str(startdate),str(enddate)).suanriqi()
        start_index=timelist.index(startdate)
        end_index=timelist.index(enddate)
        return_timelist=timelist[start_index:end_index+1]#only需要更新的时间段
        get_pe_ttm(startdate,enddate)
    
    try:
        eps_est=pd.read_csv(dir_path+'/eps_est.csv') #用于更新检查日期的指标
        last_update_day=list(eps_est.columns)[-1]  #前一次更新后的最后一天，用于确定下次更新开始的时间
        if last_update_day>=enddate:
            pass
        else:
            startdate=next_trade_date(last_update_day)
            all_timelist=get_alltimelist(str(startdate),str(enddate)).suanriqi()
            start_index=timelist.index(startdate)
            end_index=timelist.index(enddate)
            return_timelist=timelist[start_index:end_index+1]
            get_eps_est(startdate,enddate)       
    except FileNotFoundError:
        if is_trade_date(startdate)==0:          
            startdate=next_trade_date(startdate)
        else:
            startdate=startdate
        all_timelist=get_alltimelist(str(startdate),str(enddate)).suanriqi()
        start_index=timelist.index(startdate)
        end_index=timelist.index(enddate)
        return_timelist=timelist[start_index:end_index+1]#only需要更新的时间段
        get_eps_est(startdate,enddate)
    #end=time.time()
    #print((end-start)/60.0)
