# coding: utf-8

import cx_Oracle
import pandas as pd
import datetime
from datetime import timedelta
import time
import os
import numpy as np
import tushare as ts

dir_path = '/usr/datashare/fun_pq_factors'
des_path = 'D:/wuyu/untitled5/fun_pq_factors'

def getStock_ST():
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,entry_dt,remove_dt,s_type_st from wind.AShareST"
    d = pd.read_sql(sql, db)
    db.close()
    d.sort_values(by='S_INFO_WINDCODE', inplace=True)
    d.to_csv(des_path + '/ST.csv')
    return d


def getStock_IPO():
    db = cx_Oracle.connect("wind_read_only", "wind_read_only", "192.168.0.223:1521/orcl")
    sql = "select s_info_windcode,s_ipo_listdate from wind.AShareIPO"
    d = pd.read_sql(sql, db)
    db.close()
    d.sort_values(by='S_INFO_WINDCODE', inplace=True)
    d.to_csv(des_path + '/IPO.csv')
    return d

if __name__=='__main__':
    st_state = getStock_ST()
    ipo_Date = getStock_IPO()