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

def convert_sh2_snap(datafile, outfile):
    pos = datafile.rfind('/')
    if (pos == -1):
        print(True)
        indexfile = 'HisSnap.dat'  # find in current folder
    else:
        indexfile = datafile[0:pos] + 'HisSnap.dat'
    print(indexfile,datafile,outfile)

    f1 = open(datafile, 'rb')
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
            #f2.seek(idx)
            #print(code,idx)
            snap = f2.read(SH2_SNAP_SIZE)
            #print(len(snap))
            tup = struct.unpack("=Q6I8s4Q4I4Q16I10Q10I50Q10I10Q10I50Q2I2QI2Q", snap)
            str_snap = code
            str_snap += ',' + str(tup[0])  # datetime
            for i in range(5):
                str_snap += ',' + '{0:.3f}'.format(tup[i + 1] / 1000)  # price
            str_snap += ',' + '{0:.0f}'.format(tup[8] / 1000)  # volume
            str_snap += ',' + '{0:.2f}'.format(tup[9] / 100000)  # amount
            ins_status = tup[7]
            ins_status = ins_status[0:ins_status.find('\x00')]
            str_snap += ',' + ins_status  # instrument status
            for i in range(10):
                str_snap += ',' + '{0:.3f}'.format(tup[i + 26] / 1000)  # bidpx
            for i in range(10):
                str_snap += ',' + '{0:.0f}'.format(tup[i + 36] / 1000)  # bidorderqty
            for i in range(10):
                str_snap += ',' + str(tup[i + 46])  # bidnumorders
            for i in range(50):
                str_snap += ',' + '{0:.0f}'.format(tup[i + 56] / 1000)  # bidorders
            for i in range(10):
                str_snap += ',' + '{0:.3f}'.format(tup[i + 106] / 1000)  # offerpx
            for i in range(10):
                str_snap += ',' + '{0:.0f}'.format(tup[i + 116] / 1000)  # offerorderqty
            for i in range(10):
                str_snap += ',' + str(tup[i + 126])  # offernumorders
            for i in range(50):
                str_snap += ',' + '{0:.0f}'.format(tup[i + 136] / 1000)  # offerorders
            str_snap += ',' + str(tup[6])  # numtrades
            str_snap += ',' + '{0:.3f}'.format(tup[186] / 1000)  # IOPV
            for i in range(2):
                str_snap += ',' + '{0:.0f}'.format(tup[i + 10] / 1000)  # totalbidqty, totalofferqty
            for i in range(2):
                str_snap += ',' + '{0:.3f}'.format(tup[i + 12] / 1000)  # weightedavgbidpx, weightedavgofferpx
            for i in range(6):
                str_snap += ',' + str(tup[i + 20])  # totalbidnum, totaloffernum, bidtrademaxduration, offertrademaxduration, numbidorders, numofferorders
            str_snap += ',' + str(tup[14]) # withdrawbuynumber
            str_snap += ',' + '{0:.0f}'.format(tup[16] / 1000) # withdrawbuyamount
            str_snap += ',' + '{0:.2f}'.format(tup[17] / 100000) # withdrawbuymoney
            str_snap += ',' + str(tup[15]) # withdrawsellnumber
            str_snap += ',' + '{0:.0f}'.format(tup[18] / 1000) # withdrawsellamount
            str_snap += ',' + '{0:.2f}'.format(tup[19] / 100000) # withdrawsellmoney
            str_snap += ',' + str(tup[187]) # etfbuynumber
            str_snap += ',' + '{0:.0f}'.format(tup[188] / 1000) # etfbuyamount
            str_snap += ',' + '{0:.2f}'.format(tup[189] / 100000) # etfbuymoney
            str_snap += ',' + str(tup[190]) # etfsellnumber
            str_snap += ',' + '{0:.0f}'.format(tup[191] / 1000) # etfsellamount
            str_snap += ',' + '{0:.2f}'.format(tup[192] / 100000) # etfsellmoney
            str_snap += '\n'
            f3.write(str_snap)
    finally:
        f1.close()
        f2.close()
        f3.close()


if __name__=='__main__':
    datalog= os.getcwd()
    print(datalog)
    datafile=datalog+'\\HisSnap.dat'
    outfile=datalog+'\\HisSnap.csv'
    convert_sh2_snap(datafile,outfile)