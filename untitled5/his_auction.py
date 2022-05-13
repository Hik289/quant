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

def convert_sh2_auction(datafile, outfile):
    pos = datafile.rfind('/')
    if (pos == -1):
        indexfile = 'HisAuction.dat'  # find in current folder
    else:
        indexfile = datafile[0:pos] + '/HisAuction.dat'

    f1 = open(indexfile, 'rb')
    f2 = open(datafile, 'rb')
    f3 = open(outfile, 'w')
    try:
        while True:
            s = f1.read(IDX_SIZE)
            if not s:
                break
            code, idx = struct.unpack("16sQ", s)
            pos0 = code.find('\x00')
            code = code[0:pos0]
            f2.seek(idx)
            print(code,idx)
            snap = f2.read(SH2_AUCTION_SIZE)
            #print(snap)
            '''if len(snap)!= SH2_AUCTION_SIZE:
                print("ppppp")
                continue'''
            tup = struct.unpack("=QI2Q8s", snap)
            str_snap = code
            str_snap += ',' + str(tup[0])  # datetime
            str_snap += ',' + '{0:.3f}'.format(tup[1] / 1000)  # price
            str_snap += ',' + '{0:.0f}'.format(tup[2] / 1000)  # virtualauctionqty
            str_snap += ',' + '{0:.0f}'.format(tup[3] / 1000)  # leaveqty
            side = tup[4]
            side = side[0:side.find('\x00')]
            str_snap += ',' + side  # side
            str_snap += '\n'
            f3.write(str_snap)
    finally:
        f1.close()
        f2.close()
        f3.close()

if __name__=='__main__':
    datalog= os.getcwd()
    print(datalog)
    datafile=datalog+'/HisAuction.dat'
    outfile=datalog+'/HisAuction.csv'
    convert_sh2_auction(datafile,outfile)