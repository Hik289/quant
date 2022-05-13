# -*- coding: UTF-8 -*-

from __future__ import division
import sys
import struct
import os

IDX_SIZE = 24
SH1_SNAP_SIZE = 188
SH2_SNAP_SIZE = 1308
SH2_TICK_SIZE = 40
SH2_AUCTION_SIZE = 36
SHO_SNAP_SIZE = 236
HEAD_SIZE = 52
INDEX_SIZE = 32
KLINE_SIZE = 52


def convert_sh2_tick(datafile, outfile):

    f1 = open(datafile, 'rb')
    f3 = open(outfile, 'w')
    try:
        s = f1.read(HEAD_SIZE)
        tup_head = struct.unpack("=16sQ7I", s)
        for i in range(tup_head[8]): # code count
            f1.seek(HEAD_SIZE + tup_head[3] + INDEX_SIZE * i)  # seek to the beginning of index
            index = f1.read(INDEX_SIZE)
            tup_index = struct.unpack("=16s2Q", index)
            code = tup_index[0]
            code = code[0:code.find('\x00')]
            pos = tup_index[1]
            while pos < tup_index[2]:
                f1.seek(pos)
                snap = f1.read(SH2_TICK_SIZE)
                tup = struct.unpack("=Q2I3Q", snap)
                str_snap = code
                if code[0]!='6':
                    pos += SH2_TICK_SIZE
                    continue
                str_snap += ',' + str(tup[0])  # datetime
                str_snap += ',' + '{0:.3f}'.format(tup[1] / 1000)  # tradeprice
                str_snap += ',' + '{0:.0f}'.format(tup[2] / 1000)  # tradeqty
                str_snap += ',' + '{0:.2f}'.format(tup[3] / 100000)  # tradeamount
                str_snap += ',' + str(tup[4]) # buyno
                str_snap += ',' + str(tup[5]) # sellno

                print(str_snap)
                str_snap += '\n'
                f3.write(str_snap)
                pos += SH2_TICK_SIZE
    finally:
        f1.close()
        f3.close()


if __name__=='__main__':
    datalog= os.getcwd()
    print(datalog)
    datafile=datalog+'\\HisTick.dat'
    outfile=datalog+'\\HisTick.csv'
    convert_sh2_tick(datafile,outfile)