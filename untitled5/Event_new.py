# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:40:56 2019

Event residual

"""

import os
import pandas as pd
import numpy as np
import cx_Oracle
import datetime

#事件时间  20050217--20181210
x='20050217'
y='20181210'
x_str = str(x)
y_str = str(y)
db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl",encoding='utf-8')

#读入残差
residual = pd.read_csv('NewResidual.csv', header = 0,
                       index_col = 0)

#事件信息
#1.业绩预增
sql = "SELECT s_info_windcode, s_profitnotice_date FROM wind.AShareProfitNotice WHERE s_profitnotice_date>=%s AND s_profitnotice_date<=%s AND s_profitnotice_style in ('454004000','454003000','454010000') " %(x_str,y_str)
PN = pd.read_sql(sql,db)  
PNindex = list((PN['S_INFO_WINDCODE']+'_'+PN['S_PROFITNOTICE_DATE']).unique())  #39763个事件
AR = PickRsdl(PNindex, residual)

#2.业绩预亏
#业绩预亏(首亏、略减、续亏、预减)  --AShareProfitNotice表
sql = "SELECT s_info_windcode, s_profitnotice_date FROM wind.AShareProfitNotice WHERE s_profitnotice_date>=%s AND s_profitnotice_date<=%s AND s_profitnotice_style in (454002000,454006000,454007000,454009000) " %(x_str,y_str)
PNloss = pd.read_sql(sql,db)  #个事件
PNLSindex = list((PNloss['S_INFO_WINDCODE']+'_'+PNloss['S_PROFITNOTICE_DATE']).unique())

#3.股东增持（Insider Trade)
sql = "SELECT s_info_windcode, Actual_ann_dt FROM wind.AShareInsiderTrade WHERE Actual_ann_dt>=%s AND Actual_ann_dt<=%s AND change_volume>0 " %(x_str,y_str)
isdr = pd.read_sql(sql,db) 
test =  isdr['S_INFO_WINDCODE']+'_'+isdr['ACTUAL_ANN_DT']
ISDRindex = list((isdr['S_INFO_WINDCODE']+'_'+isdr['ACTUAL_ANN_DT']).unique())  
AR = PickRsdl(ISDRindex, residual)
AR.to_csv('InsiderTradeUP.csv')
AR.mean().to_csv('InsiderTradeUPmean.csv')
#股东增持version2: Insider+mjr[1/3]前半段 <20131121
z='20131121'
z_str=str(z)
sql = "SELECT s_info_windcode,ann_dt FROM wind.AShareMjrHolderTrade WHERE ann_dt>=%s AND ann_dt<%s AND TRANSACT_TYPE='增持' AND holder_type in (1,3)" %(x_str,z_str)
MJR = pd.read_sql(sql,db)  #MJR前半段增持事件
MJRindex = list((MJR['S_INFO_WINDCODE']+'_'+MJR['ANN_DT']).unique())  #转成list
ISDRall = list(pd.Series(ISDRindex + MJRindex).unique())
AR = PickRsdl(ISDRall, residual)
AR.to_csv('exeUP.csv')
AR.mean().to_csv('exeUPmean.csv')
#股东增持version3:纯mjr表 效果不好
#股东增持version4:mjr[2] --纯公司股东 
sql = "SELECT s_info_windcode,ann_dt FROM wind.AShareMjrHolderTrade WHERE ann_dt>=%s AND ann_dt<=%s AND TRANSACT_TYPE='增持' AND holder_type in (2)" %(x_str,y_str)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!
MJR_cpny = pd.read_sql(sql,db)  #MJR_company前半段增持事件
MJRCOM = list((MJR_cpny['S_INFO_WINDCODE']+'_'+MJR_cpny['ANN_DT']).unique())  #转成list

AR = PickRsdl(MJRCOM, residual)
AR.to_csv('comUP.csv')
AR.mean().to_csv('comUPmean.csv')

#!!!!!!!!!!!!!!!!!!!!!!!!!!!

#4.股权激励
sql = "SELECT s_info_windcode,PREPLAN_ANN_DATE FROM wind.AShareIncDescription WHERE PREPLAN_ANN_DATE>=%s AND PREPLAN_ANN_DATE<=%s" %(x_str,y_str)
inc = pd.read_sql(sql,db)
incindex = list((inc['S_INFO_WINDCODE']+'_'+inc['PREPLAN_ANN_DATE']).unique())  #余下2062个事件
AR = PickRsdl(incindex, residual)
AR.to_csv('IncentiveAR.csv')
AR.mean().to_csv('IncentiveMean.csv')

#5.投资者调研
#投资者调研--机构调研活动
sql = "SELECT S_INFO_WINDCODE,S_SURVEYDATE FROM wind.AshareISActivity WHERE s_surveydate>=%s AND s_surveydate<=%s"%(x_str,y_str)
srvy = pd.read_sql(sql,db)
srvyindex = list((srvy['S_INFO_WINDCODE']+'_'+srvy['S_SURVEYDATE']).unique())
AR = PickRsdl(srvyindex, residual)
AR.to_csv('SurveyAR.csv')
AR.mean().to_csv('SurveyMean.csv')


