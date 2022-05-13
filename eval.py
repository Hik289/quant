#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from os import walk

root_path = r'/home/xysong/PRODUCT'
import sys

sys.path.append(root_path)
from common_lib import QLLIB

path = r'/home/xysong/PRODUCT/workspace_intern/data/advanced_result'
data_path = r'/home/xysong/PRODUCT/workspace_data'
f = []
for (dirpath, dirnames, filenames) in walk(path):
    f.extend(filenames)
    break
q = []
for filename in f:
    data = QLLIB.read_from_selflib(data_name=filename[:-4], data_path=path)
    q.append((filename[:-4], data))

for i in q:
    print('polt IC', i[0])
    QLLIB.count_ic_free('a', i[1], \
                        data_path=data_path, begin_date='20180705', end_date='20190701', \
                        price='vwap_p', cycle=1, rank_ic=True, add_info=None)

'''    
    QLLIB.count_pnl_free('a',i[1],\
                         data_path = data_path,begin_date = '20180705',end_date = '20190701',\
                         price = 'vwap_p',amount_limit = 3000,cycle =1,equal_weight = True,top_num = 500,add_info = None)


    QLLIB.count_ic_level(i[1],\
                         data_path = data_path,begin_date = '20180705',end_date = '20190701',\
                         price = 'vwap_p',cycle = 1,rank_ic = False,add_info = None)
'''



























