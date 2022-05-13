# -*- coding: UTF-8 -*-

import sys
import struct
import os

HEAD_SIZE = 52
INDEX_SIZE = 32
SH2_SNAP_SIZE = 1308


def convert_sh2_snap(datafile,outfile):
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
                snap = f1.read(SH2_SNAP_SIZE)
                tup = struct.unpack("=Q6I8s4Q4I4Q16I10Q10I50Q10I10Q10I50Q2I2QI2Q", snap)
                str_snap = code+'.SZ'
                if code[0]!='3' or code[0]!= '0':
                    pos += SH2_SNAP_SIZE
                    continue
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

                print(str_snap)
                str_snap += '\n'
                f3.write(str_snap)
                pos += SH2_SNAP_SIZE
    finally:
        f1.close()
        f3.close()


if __name__=='__main__':
    datalog= os.getcwd()
    print(datalog)
    datafile=datalog+'\\HisSnap.dat'
    outfile='~/data/HisSnap.csv'
    convert_sh2_snap(datafile,outfile)