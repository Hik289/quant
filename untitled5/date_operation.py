import pandas as pd
import datetime
from datetime import timedelta
import tushare as ts

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