# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 10:15:50 2018

@author: 黄德金
"""
import pandas as pd
import cx_Oracle
import numpy as np

des_path='/usr/intern/wufei/fun_factor'
VOLUME=pd.read_csv('/usr/intern/wufei/fun_factor/volume.csv',index_col='S_INFO_WINDCODE')
timelist=VOLUME.columns
stocklist=VOLUME.index
def get_est_bpsavg(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,est_dt,s_est_avgbps from wind.AShareConsensusData where est_dt>=%s AND est_dt<=%s order by est_dt" % (
    x_str, y_str)
    d = pd.read_sql(sql,db)
    d=pd.DataFrame(d)
    d = d.set_index('S_INFO_WINDCODE')
    d=d.groupby(['S_INFO_WINDCODE', 'EST_DT']).mean()
    d = d.unstack(level=0).T
    d.index = d.index.droplevel(0)
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    g=g.fillna(method='pad',axis=1)
    g.to_csv(des_path+'/est_bpsavg.csv')
def get_price(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,trade_dt,s_dq_close from wind.AShareEODPrices where trade_dt>=%s AND trade_dt<=%s order by trade_dt" % (
    x_str, y_str)
    d = pd.read_sql(sql,db)
    d=pd.DataFrame(d)
    d = d.set_index(['TRADE_DT', 'S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    d=d.sort_index()
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    g.to_csv(des_path+'/price.csv')
x=20091220
y=20180831
get_est_bpsavg(x,y)
get_price(x,y)
price=pd.read_csv('/usr/intern/wufei/fun_factor/price.csv',index_col='S_INFO_WINDCODE')
est_bpsavg=pd.read_csv('/usr/intern/wufei/fun_factor/est_bpsavg.csv',index_col='S_INFO_WINDCODE')
est_pb=price/est_bpsavg
est_pb.to_csv(des_path+'/alpha2.csv')

'''
def get_RG(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,ann_dt,report_period,s_fa_yoy_or from wind.AShareFinancialIndicator where ann_dt>=%s AND ann_dt<=%s order by s_info_windcode" % (x_str,y_str)
    d=pd.read_sql(sql,db)
    d=pd.DataFrame(d)
    a=d.iloc[:,3]
    a=a.dropna()
    b=d.iloc[:,0:3]
    d=pd.concat([a,b],axis=1,join='inner')
    d=d.sort_values(by=['S_INFO_WINDCODE','ANN_DT','REPORT_PERIOD'],ascending=False)
    d=d.drop_duplicates(['S_INFO_WINDCODE','ANN_DT'],keep='first')
    d=d.drop('REPORT_PERIOD',axis=1)
    d=d.set_index(['ANN_DT','S_INFO_WINDCODE'])
    d=d.stack(level=0)
    d=d.unstack(level=0)
    d.index=d.index.droplevel(level=1)
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    g=g.fillna(method='pad',axis=1)
    g.to_csv(des_path+'/alpha23.csv')

def get_RGQ(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,ann_dt,report_period,s_qfa_yoysales from wind.AShareFinancialIndicator where ann_dt>=%s AND ann_dt<=%s order by s_info_windcode" % (x_str,y_str)
    d=pd.read_sql(sql,db)
    d=pd.DataFrame(d)
    a=d.iloc[:,3]
    a=a.dropna()
    b=d.iloc[:,0:3]
    d=pd.concat([a,b],axis=1,join='inner')
    d=d.sort_values(by=['S_INFO_WINDCODE','ANN_DT','REPORT_PERIOD'],ascending=False)
    d=d.drop_duplicates(['S_INFO_WINDCODE','ANN_DT'],keep='first')
    d=d.drop('REPORT_PERIOD',axis=1)
    d=d.set_index(['ANN_DT','S_INFO_WINDCODE'])
    d=d.stack(level=0)
    d=d.unstack(level=0)
    d.index=d.index.droplevel(level=1)
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    g=g.fillna(method='pad',axis=1)
    g.to_csv(des_path+'/alpha24.csv')

def get_RGQ2(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,ann_dt,report_period,s_qfa_cgrsales from wind.AShareFinancialIndicator where ann_dt>=%s AND ann_dt<=%s order by s_info_windcode" % (x_str,y_str)
    d=pd.read_sql(sql,db)
    d=pd.DataFrame(d)
    a=d.iloc[:,3]
    a=a.dropna()
    b=d.iloc[:,0:3]
    d=pd.concat([a,b],axis=1,join='inner')
    d=d.sort_values(by=['S_INFO_WINDCODE','ANN_DT','REPORT_PERIOD'],ascending=False)
    d=d.drop_duplicates(['S_INFO_WINDCODE','ANN_DT'],keep='first')
    d=d.drop('REPORT_PERIOD',axis=1)
    d=d.set_index(['ANN_DT','S_INFO_WINDCODE'])
    d=d.stack(level=0)
    d=d.unstack(level=0)
    d.index=d.index.droplevel(level=1)
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    g=g.fillna(method='pad',axis=1)
    g.to_csv(des_path+'/alpha25.csv')
    
def get_NPGQ(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,ann_dt,report_period,s_qfa_yoyprofit from wind.AShareFinancialIndicator where ann_dt>=%s AND ann_dt<=%s order by s_info_windcode" % (x_str,y_str)
    d=pd.read_sql(sql,db)
    d=pd.DataFrame(d)
    a=d.iloc[:,3]
    a=a.dropna()
    b=d.iloc[:,0:3]
    d=pd.concat([a,b],axis=1,join='inner')
    d=d.sort_values(by=['S_INFO_WINDCODE','ANN_DT','REPORT_PERIOD'],ascending=False)
    d=d.drop_duplicates(['S_INFO_WINDCODE','ANN_DT'],keep='first')
    d=d.drop('REPORT_PERIOD',axis=1)
    d=d.set_index(['ANN_DT','S_INFO_WINDCODE'])
    d=d.stack(level=0)
    d=d.unstack(level=0)
    d.index=d.index.droplevel(level=1)
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    g=g.fillna(method='pad',axis=1)
    g.to_csv(des_path+'/alpha26.csv')

def get_NPGQ2(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,ann_dt,report_period,s_qfa_cgrprofit from wind.AShareFinancialIndicator where ann_dt>=%s AND ann_dt<=%s order by s_info_windcode" % (x_str,y_str)
    d=pd.read_sql(sql,db)
    d=pd.DataFrame(d)
    a=d.iloc[:,3]
    a=a.dropna()
    b=d.iloc[:,0:3]
    d=pd.concat([a,b],axis=1,join='inner')
    d=d.sort_values(by=['S_INFO_WINDCODE','ANN_DT','REPORT_PERIOD'],ascending=False)
    d=d.drop_duplicates(['S_INFO_WINDCODE','ANN_DT'],keep='first')
    d=d.drop('REPORT_PERIOD',axis=1)
    d=d.set_index(['ANN_DT','S_INFO_WINDCODE'])
    d=d.stack(level=0)
    d=d.unstack(level=0)
    d.index=d.index.droplevel(level=1)
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    g=g.fillna(method='pad',axis=1)
    g.to_csv(des_path+'/alpha27.csv')

def get_OCFG(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,ann_dt,report_period,s_fa_yoyocf from wind.AShareFinancialIndicator where ann_dt>=%s AND ann_dt<=%s order by s_info_windcode" % (x_str,y_str)
    d=pd.read_sql(sql,db)
    d=pd.DataFrame(d)
    a=d.iloc[:,3]
    a=a.dropna()
    b=d.iloc[:,0:3]
    d=pd.concat([a,b],axis=1,join='inner')
    d=d.sort_values(by=['S_INFO_WINDCODE','ANN_DT','REPORT_PERIOD'],ascending=False)
    d=d.drop_duplicates(['S_INFO_WINDCODE','ANN_DT'],keep='first')
    d=d.drop('REPORT_PERIOD',axis=1)
    d=d.set_index(['ANN_DT','S_INFO_WINDCODE'])
    d=d.stack(level=0)
    d=d.unstack(level=0)
    d.index=d.index.droplevel(level=1)
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    g=g.fillna(method='pad',axis=1)
    g.to_csv(des_path+'/alpha28.csv')

def get_ROEG(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,ann_dt,report_period,s_fa_yoyroe from wind.AShareFinancialIndicator where ann_dt>=%s AND ann_dt<=%s order by s_info_windcode" % (x_str,y_str)
    d=pd.read_sql(sql,db)
    d=pd.DataFrame(d)
    a=d.iloc[:,3]
    a=a.dropna()
    b=d.iloc[:,0:3]
    d=pd.concat([a,b],axis=1,join='inner')
    d=d.sort_values(by=['S_INFO_WINDCODE','ANN_DT','REPORT_PERIOD'],ascending=False)
    d=d.drop_duplicates(['S_INFO_WINDCODE','ANN_DT'],keep='first')
    d=d.drop('REPORT_PERIOD',axis=1)
    d=d.set_index(['ANN_DT','S_INFO_WINDCODE'])
    d=d.stack(level=0)
    d=d.unstack(level=0)
    d.index=d.index.droplevel(level=1)
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    g=g.fillna(method='pad',axis=1)
    g.to_csv(des_path+'/alpha29.csv')

x=20091220
y=20180831
get_RG(x,y)
get_RGQ(x,y)
get_RGQ2(x,y)
get_NPGQ(x,y)
get_NPGQ2(x,y)
get_OCFG(x,y)
get_ROEG(x,y)


def get_mv_total(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,trade_dt,s_val_mv from wind.AShareEODDerivativeIndicator where trade_dt>=%s AND trade_dt<=%s order by trade_dt" % (
    x_str, y_str)
    d = pd.read_sql(sql,db)
    d = d.set_index(['TRADE_DT', 'S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    d=d.sort_index()
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    g.to_csv(des_path+'/mv_total.csv')
    
def get_mv_circulation(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,trade_dt,s_dq_mv from wind.AShareEODDerivativeIndicator where trade_dt>=%s AND trade_dt<=%s order by trade_dt" % (
    x_str, y_str)
    d = pd.read_sql(sql,db)
    d = d.set_index(['TRADE_DT', 'S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    d=d.sort_index()
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    g.to_csv(des_path+'/mv_circulation.csv')

def get_price(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,trade_dt,s_dq_close from wind.AShareEODPrices where trade_dt>=%s AND trade_dt<=%s order by trade_dt" % (
    x_str, y_str)
    d = pd.read_sql(sql,db)
    d=pd.DataFrame(d)
    d = d.set_index(['TRADE_DT', 'S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    d=d.sort_index()
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    g.to_csv(des_path+'/price.csv')


### 总资产净利润
def get_roa(x,y):
    x_str=str(x)
    y_str=str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,ann_dt,report_period,s_fa_roa from wind.AShareFinancialIndicator where ann_dt>=%s AND ann_dt<=%s order by ann_dt" % (x_str,y_str)
    d=pd.read_sql(sql,db)
    d=pd.DataFrame(d)
    d=d.sort_values(by=['S_INFO_WINDCODE','ANN_DT','REPORT_PERIOD'],ascending=False)
    d=d.drop_duplicates(['S_INFO_WINDCODE','ANN_DT'],keep='first')
    d=d.drop('REPORT_PERIOD',axis=1)
    d = d.set_index(['ANN_DT','S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    d=d.sort_index()
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    g=g.fillna(method='pad',axis=1)
    g.to_csv(des_path+'/roa.csv')


### 总资产报酬率
def get_roa2(x,y):
    x_str=str(x)
    y_str=str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,ann_dt,report_period,s_fa_roa2 from wind.AShareFinancialIndicator where ann_dt>=%s AND ann_dt<=%s order by ann_dt" % (x_str,y_str)
    d=pd.read_sql(sql,db)
    d=pd.DataFrame(d)
    d=d.sort_values(by=['S_INFO_WINDCODE','ANN_DT','REPORT_PERIOD'],ascending=False)
    d=d.drop_duplicates(['S_INFO_WINDCODE','ANN_DT'],keep='first')
    d=d.drop('REPORT_PERIOD',axis=1)
    d = d.set_index(['ANN_DT','S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    d=d.sort_index()
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    g=g.fillna(method='pad',axis=1)
    g.to_csv(des_path+'/roa2.csv')
    
def get_roe(x,y):
    x_str=str(x)
    y_str=str(y)
    db=cx_Oracle.connect("wind_read_only","wind_read_only","192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,ann_dt,report_period,s_fa_roe from wind.AShareFinancialIndicator where ann_dt>=%s AND ann_dt<=%s order by ann_dt" % (x_str,y_str)
    d=pd.read_sql(sql,db)
    d=pd.DataFrame(d)
    d=d.sort_values(by=['S_INFO_WINDCODE','ANN_DT','REPORT_PERIOD'],ascending=False)
    d=d.drop_duplicates(['S_INFO_WINDCODE','ANN_DT'],keep='first')
    d=d.drop('REPORT_PERIOD',axis=1)
    d = d.set_index(['ANN_DT','S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    d=d.sort_index()
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    g=g.fillna(method='pad',axis=1)
    g.to_csv(des_path+'/roe.csv')

def get_roe_ded(x,y):
    x_str=str(x)
    y_str=str(y)
    db=cx_Oracle.connect("wind_read_only","wind_read_only","192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,ann_dt,report_period,s_fa_roe_deducted from wind.AShareFinancialIndicator where ann_dt>=%s AND ann_dt<=%s order by ann_dt" % (x_str,y_str)
    d=pd.read_sql(sql,db)
    d=pd.DataFrame(d)
    d=d.sort_values(by=['S_INFO_WINDCODE','ANN_DT','REPORT_PERIOD'],ascending=False)
    d=d.drop_duplicates(['S_INFO_WINDCODE','ANN_DT'],keep='first')
    d=d.drop('REPORT_PERIOD',axis=1)
    d = d.set_index(['ANN_DT','S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    d=d.sort_index()
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    g=g.fillna(method='pad',axis=1)
    g.to_csv(des_path+'/roe_ded.csv')

def get_assetstoequity(x,y):
    x_str=str(x)
    y_str=str(y)
    db=cx_Oracle.connect("wind_read_only","wind_read_only","192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,ann_dt,report_period,s_fa_assetstoequity from wind.AShareFinancialIndicator where ann_dt>=%s AND ann_dt<=%s order by ann_dt" % (x_str,y_str)
    d=pd.read_sql(sql,db)
    d=pd.DataFrame(d)
    d=d.sort_values(by=['S_INFO_WINDCODE','ANN_DT','REPORT_PERIOD'],ascending=False)
    d=d.drop_duplicates(['S_INFO_WINDCODE','ANN_DT'],keep='first')
    d=d.drop('REPORT_PERIOD',axis=1)
    d = d.set_index(['ANN_DT','S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    d=d.sort_index()
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    g=g.fillna(method='pad',axis=1)
    g.to_csv(des_path+'/assetstoequity.csv')
    
def get_interestdebt(x,y):
    x_str=str(x)
    y_str=str(y)
    db=cx_Oracle.connect("wind_read_only","wind_read_only","192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,ann_dt,report_period,s_fa_interestdebt from wind.AShareFinancialIndicator where ann_dt>=%s AND ann_dt<=%s order by ann_dt" % (x_str,y_str)
    d=pd.read_sql(sql,db)
    d=pd.DataFrame(d)
    d=d.sort_values(by=['S_INFO_WINDCODE','ANN_DT','REPORT_PERIOD'],ascending=False)
    d=d.drop_duplicates(['S_INFO_WINDCODE','ANN_DT'],keep='first')
    d=d.drop('REPORT_PERIOD',axis=1)
    d = d.set_index(['ANN_DT','S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    d=d.sort_index()
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    g=g.fillna(method='pad',axis=1)
    g.to_csv(des_path+'/interestdebt.csv')
    
def get_netprofitmargin(x,y):
    x_str=str(x)
    y_str=str(y)
    db=cx_Oracle.connect("wind_read_only","wind_read_only","192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,ann_dt,report_period,s_fa_netprofitmargin from wind.AShareFinancialIndicator where ann_dt>=%s AND ann_dt<=%s order by ann_dt" % (x_str,y_str)
    d=pd.read_sql(sql,db)
    d=pd.DataFrame(d)
    d=d.sort_values(by=['S_INFO_WINDCODE','ANN_DT','REPORT_PERIOD'],ascending=False)
    d=d.drop_duplicates(['S_INFO_WINDCODE','ANN_DT'],keep='first')
    d=d.drop('REPORT_PERIOD',axis=1)
    d = d.set_index(['ANN_DT','S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    d=d.sort_index()
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    g=g.fillna(method='pad',axis=1)
    g.to_csv(des_path+'/netprofitmargin.csv')

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
    d=d.sort_index()
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    g.to_csv(des_path+'/pe_ttm.csv')

def get_pe(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,trade_dt,s_val_pe from wind.AShareEODDerivativeIndicator where trade_dt>=%s AND trade_dt<=%s order by trade_dt" % (
    x_str, y_str)
    d = pd.read_sql(sql,db)
    d = d.set_index(['TRADE_DT', 'S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    d=d.sort_index()
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    g.to_csv(des_path+'/pe.csv')

def get_est_epsavg(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,est_dt,eps_avg from wind.AShareConsensusData where est_dt>=%s AND est_dt<=%s order by est_dt" % (
    x_str, y_str)
    d = pd.read_sql(sql,db)
    d=pd.DataFrame(d)
    d = d.set_index('S_INFO_WINDCODE')
    d=d.groupby(['S_INFO_WINDCODE', 'EST_DT']).mean()
    d = d.unstack(level=0).T
    d.index = d.index.droplevel(0)
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    g=g.fillna(method='pad',axis=1)
    g.to_csv(des_path+'/est_epsavg.csv')



    
def get_ps(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,trade_dt,s_val_ps from wind.AShareEODDerivativeIndicator where trade_dt>=%s AND trade_dt<=%s order by trade_dt" % (
    x_str, y_str)
    d = pd.read_sql(sql,db)
    d = d.set_index(['TRADE_DT', 'S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    d=d.sort_index()
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    d.to_csv(des_path+'/ps.csv')
    
def get_ps_ttm(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,trade_dt,s_val_ps_ttm from wind.AShareEODDerivativeIndicator where trade_dt>=%s AND trade_dt<=%s order by trade_dt" % (
    x_str, y_str)
    d = pd.read_sql(sql,db)
    d = d.set_index(['TRADE_DT', 'S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    d=d.sort_index()
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    g.to_csv(des_path+'/ps_ttm.csv')

def get_pcf_ocf(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,trade_dt,s_val_pcf_ocf from wind.AShareEODDerivativeIndicator where trade_dt>=%s AND trade_dt<=%s order by trade_dt" % (
    x_str, y_str)
    d = pd.read_sql(sql,db)
    d = d.set_index(['TRADE_DT', 'S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    d=d.sort_index()
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    g.to_csv(des_path+'/pcf_ocf.csv')

def get_pcf_ocfttm(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,trade_dt,s_val_pcf_ocfttm from wind.AShareEODDerivativeIndicator where trade_dt>=%s AND trade_dt<=%s order by trade_dt" % (
    x_str, y_str)
    d = pd.read_sql(sql,db)
    d = d.set_index(['TRADE_DT', 'S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    d=d.sort_index()
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    d.to_csv(des_path+'/pcf_ocfttm.csv')

def get_pcf_ncf(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,trade_dt,s_val_pcf_ncf from wind.AShareEODDerivativeIndicator where trade_dt>=%s AND trade_dt<=%s order by trade_dt" % (
    x_str, y_str)
    d = pd.read_sql(sql,db)
    d = d.set_index(['TRADE_DT', 'S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    d=d.sort_index()
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    g.to_csv(des_path+'/pcf_ncf.csv')
    
def get_pcf_ncfttm(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,trade_dt,s_val_pcf_ncfttm from wind.AShareEODDerivativeIndicator where trade_dt>=%s AND trade_dt<=%s order by trade_dt" % (
    x_str, y_str)
    d = pd.read_sql(sql,db)
    d = d.set_index(['TRADE_DT', 'S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    d=d.sort_index()
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    g.to_csv(des_path+'/pcf_ncfttm.csv')
    
def get_pb(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,trade_dt,s_val_pb_new from wind.AShareEODDerivativeIndicator where trade_dt>=%s AND trade_dt<=%s order by trade_dt" % (
    x_str, y_str)
    d = pd.read_sql(sql,db)
    d = d.set_index(['TRADE_DT', 'S_INFO_WINDCODE'])
    d = d.stack(level=0)
    d = d.unstack(level=0)
    db.close()
    d.index = d.index.droplevel(1)
    d=d.sort_index()
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    g.to_csv(des_path+'/pb.csv')

def get_est_pb(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,est_dt,s_est_pb from wind.AShareEarningEst where est_dt>=%s AND est_dt<=%s order by est_dt" % (
    x_str, y_str)
    d = pd.read_sql(sql,db)
    d=pd.DataFrame(d)
    d = d.set_index('S_INFO_WINDCODE')
    d=d.groupby(['S_INFO_WINDCODE', 'EST_DT']).mean()
    d = d.unstack(level=0).T
    d.index = d.index.droplevel(0)
    d=d.fillna(method='pad',axis=1)
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    g.to_csv(des_path+'/est_pb.csv')

    
def get_est_EVEBITDA(x,y):
    x_str = str(x)
    y_str = str(y)
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,est_dt,s_est_EVEBITDA from wind.AShareEarningEst where est_dt>=%s AND est_dt<=%s order by est_dt" % (
    x_str, y_str)
    d = pd.read_sql(sql,db)
    d=pd.DataFrame(d)
    d = d.set_index('S_INFO_WINDCODE')
    d=d.groupby(['S_INFO_WINDCODE', 'EST_DT']).mean()
    d = d.unstack(level=0).T
    d.index = d.index.droplevel(0)
    d=d.fillna(method='pad',axis=1)
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    g.to_csv(des_path+'/est_EVEBITDA.csv')

def get_main_businc(x,y):
    x_str=str(x)
    y_str=str(y)
    db=cx_Oracle.connect("wind_read_only","wind_read_only","192.168.0.223:1521/orcl")
    sql="select s_info_windcode,est_dt,main_bus_inc_avg from wind.AShareConsensusData where est_dt>=%s AND est_dt<=%s order by est_dt" % (x_str,y_str)
    d=pd.read_sql(sql,db)
    d=pd.DataFrame(d)
    d = d.set_index('S_INFO_WINDCODE')
    d=d.groupby(['S_INFO_WINDCODE', 'EST_DT']).mean()
    d = d.unstack(level=0).T
    d.index = d.index.droplevel(0)
    d=d.fillna(method='pad',axis=1)
    g=pd.DataFrame(index=stocklist,columns=timelist)
    g.iloc[:,:]=d.iloc[:,:]
    g.to_csv(des_path+'/main_businc.csv')

x=20091220
y=20180831
get_mv_total(x,y)
get_mv_circulation(x,y)
mv_total=pd.read_csv('/usr/intern/wufei/fun_factor/mv_total.csv',index_col='S_INFO_WINDCODE')
mv_circulation=pd.read_csv('/usr/intern/wufei/fun_factor/mv_circulation.csv',index_col='S_INFO_WINDCODE')
alpha21=mv_circulation/mv_total
alpha22=np.log(mv_circulation)
alpha21.to_csv(des_path+'/alpha21.csv')
alpha22.to_csv(des_path+'/alpha22.csv')

get_roa(x,y)
get_roa(x,y)
get_roa2(x,y)
get_roe(x,y)
get_roe_ded(x,y)
get_assetstoequity(x,y)
get_interestdebt(x,y)
get_netprofitmargin(x,y)

get_pe_ttm(x,y)
get_pe(x,y)
get_est_epsavg(x,y)
get_ps(x,y)
get_ps_ttm(x,y)
get_pcf_ocf(x,y)
get_pcf_ocfttm(x,y)
get_pcf_ncf(x,y)
get_pcf_ncfttm(x,y)
get_pb(x,y)
get_est_EVEBITDA(x,y)
get_main_businc(x,y)
'''

'''
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
        d4 = pd.DataFrame(index=stock_list, columns=timelist)
        d4.loc[:, :] = d3.loc[:, :]
        #再合并，处理中间没有数据的部分
        d_new=pd.concat([d_hist,d4],axis=1,sort=True)
        d_new=d_new.fillna(method='pad', axis=1)
    else:#d为空
        #生成空表
        d2=pd.DataFrame(index=stock_list, columns=all_timelist)
        d2.index.name = 'S_INFO_WINDCODE'
        d_new = pd.concat([d_hist, d2], axis=1,sort=True)
        d_new = d_new.fillna(method='pad', axis=1)
    d_new.index.name='S_INFO_WINDCODE'
    db.close()
    d_new.to_csv(des_path+'/eps_est.csv')
    return 'eps_est done'
'''