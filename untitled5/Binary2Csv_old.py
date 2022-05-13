# -*- coding: UTF-8 -*-

from __future__ import division
import sys
import struct

IDX_SIZE = 24
SH1_SNAP_SIZE = 188
SH2_SNAP_SIZE = 1308
SH2_TICK_SIZE = 40
SH2_AUCTION_SIZE = 36
SHO_SNAP_SIZE = 236
HEAD_SIZE = 52
INDEX_SIZE = 32
KLINE_SIZE = 52


def convert_sh1_snap(datafile, outfile):
    pos = datafile.rfind('/')
    if (pos == -1):
        indexfile = 'Snap.idx'  # find in current folder
    else:
        indexfile = datafile[0:pos] + '/Snap.idx'

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
            snap = f2.read(SH1_SNAP_SIZE)
            tup = struct.unpack("=Q5I2Q10I11Q2I8s", snap)
            str_snap = str(tup[0])  # datetime
            str_snap += ',' + code
            for i in range(5):
                str_snap += ',' + '{0:.3f}'.format(tup[i + 1] / 1000)  # price
            str_snap += ',' + '{0:.0f}'.format(tup[6] / 1000)  # volume
            str_snap += ',' + '{0:.2f}'.format(tup[7] / 100000)  # amount
            for i in range(5):
                str_snap += ',' + '{0:.3f}'.format(tup[i + 8] / 1000)  # bidpx
                str_snap += ',' + str(tup[i + 18])  # bidsize
                str_snap += ',' + '{0:.3f}'.format(tup[i + 13] / 1000)  # offerpx
                str_snap += ',' + str(tup[i + 22])  # offersize
            str_snap += ',' + str(tup[28])  # numtrades
            for i in range(2):
                str_snap += ',' + '{0:.3f}'.format(tup[i + 29] / 1000)  # IOPV, NAV
            phase_code = tup[31]
            phase_code = phase_code[0:phase_code.find('\x00')]
            str_snap += ',' + phase_code + '\n'  # phasecode
            f3.write(str_snap)
    finally:
        f1.close()
        f2.close()
        f3.close()


def convert_sh2_snap(datafile, outfile):
    pos = datafile.rfind('/')
    if (pos == -1):
        indexfile = 'HisSnap.dat'  # find in current folder
    else:
        indexfile = datafile[0:pos] + '/Snap.idx'

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
            snap = f2.read(SH2_SNAP_SIZE)
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


def convert_sh2_tick(datafile, outfile):
    pos = datafile.rfind('/')
    if (pos == -1):
        indexfile = 'Tick.idx'  # find in current folder
    else:
        indexfile = datafile[0:pos] + '/Tick.idx'

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
            snap = f2.read(SH2_TICK_SIZE)
            tup = struct.unpack("=Q2I3Q", snap)
            str_snap = code
            str_snap += ',' + str(tup[0])  # datetime
            str_snap += ',' + '{0:.3f}'.format(tup[1] / 1000)  # tradeprice
            str_snap += ',' + '{0:.0f}'.format(tup[2] / 1000)  # tradeqty
            str_snap += ',' + '{0:.2f}'.format(tup[3] / 100000)  # tradeamount
            str_snap += ',' + str(tup[4]) # buyno
            str_snap += ',' + str(tup[5]) # sellno
            str_snap += '\n'
            f3.write(str_snap)
    finally:
        f1.close()
        f2.close()
        f3.close()


def convert_sh2_auction(datafile, outfile):
    pos = datafile.rfind('/')
    if (pos == -1):
        indexfile = 'Auction.idx'  # find in current folder
    else:
        indexfile = datafile[0:pos] + '/Auction.idx'

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
            snap = f2.read(SH2_AUCTION_SIZE)
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


def convert_sho_snap(datafile, outfile):
    pos = datafile.rfind('/')
    if (pos == -1):
        indexfile = 'Snap.idx'  # find in current folder
    else:
        indexfile = datafile[0:pos] + '/Snap.idx'

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
            snap = f2.read(SHO_SNAP_SIZE)
            tup = struct.unpack("=QI27Q8s", snap)
            str_snap = str(tup[0])  # datetime
            str_snap += ',' + code
            for i in range(5):
                str_snap += ',' + '{0:.4f}'.format(tup[i + 1] / 10000)  # price
            str_snap += ',' + str(tup[6])  # totallongposition
            for i in range(5):
                str_snap += ',' + '{0:.4f}'.format(tup[i + 12] / 10000)  # bidpx
                str_snap += ',' + str(tup[i + 7])  # bidsize
                str_snap += ',' + '{0:.4f}'.format(tup[i + 22] / 10000)  # offerpx
                str_snap += ',' + str(tup[i + 17])  # offersize
            str_snap += ',' + str(tup[27])  # totalvolumetrade
            str_snap += ',' + '{0:.2f}'.format(tup[28] / 100)  # totalvaluetrade
            phase_code = tup[29]
            phase_code = phase_code[0:phase_code.find('\x00')]
            str_snap += ',' + phase_code + '\n'  # phasecode
            f3.write(str_snap)
    finally:
        f1.close()
        f2.close()
        f3.close()


def convert_sh1_sh2_kline(datafile, outfile):
    f1 = open(datafile, 'rb')
    f2 = open(outfile, 'w')
    try:
        # find the index
        s = f1.read(HEAD_SIZE)
        tup_head = struct.unpack("=16sQ7I", s)
        for i in range(tup_head[8]):  # code count
            f1.seek(HEAD_SIZE + tup_head[3] + INDEX_SIZE * i)  # seek to the beginning of index
            index = f1.read(INDEX_SIZE)
            tup_index = struct.unpack("=16s2Q", index)
            code = tup_index[0]
            code = code[0:code.find('\x00')]
            pos = tup_index[1]
            while pos < tup_index[2]:
                f1.seek(pos)
                kline = f1.read(KLINE_SIZE)
                tup_kline = struct.unpack("=Q5I3Q", kline)
                str_kline = code  # code
                str_kline += ',' + str(tup_kline[0] / 1000000)  # datetime
                for j in range(5):
                    str_kline += ',' + '{0:.3f}'.format(tup_kline[j + 1] / 1000)  # price
                str_kline += ',' + '{0:.0f}'.format(tup_kline[6] / 1000)  # volume
                str_kline += ',' + '{0:.2f}'.format(tup_kline[7] / 100000)  # amount
                str_kline += ',' + '{0:.3f}'.format(tup_kline[8] / 1000)  # IOPV
                str_kline += '\n'
                f2.write(str_kline)
                pos += KLINE_SIZE
    finally:
        f1.close()
        f2.close()


def convert_sho_kline(datafile, outfile):
    f1 = open(datafile, 'rb')
    f2 = open(outfile, 'w')
    try:
        # find the index
        s = f1.read(HEAD_SIZE)
        tup_head = struct.unpack("16sQ7I", s)
        for i in range(tup_head[8]):  # code count
            f1.seek(HEAD_SIZE + tup_head[3] + INDEX_SIZE * i)  # seek to the beginning of index
            index = f1.read(INDEX_SIZE)
            tup_index = struct.unpack("16s2Q", index)
            code = tup_index[0]
            code = code[0:code.find('\x00')]
            pos = tup_index[1]
            while pos < tup_index[2]:
                f1.seek(pos)
                kline = f1.read(KLINE_SIZE)
                tup_kline = struct.unpack("11Q", kline)
                str_kline = code  # code
                str_kline += ',' + str(tup_kline[0])  # datetime
                for j in range(5):
                    str_kline += ',' + '{0:.4f}'.format(tup_kline[j + 1] / 10000)  # price
                str_kline += ',' + str(tup_kline[6])  # volume
                str_kline += ',' + '{0:.2f}'.format(tup_kline[7] / 100)  # amount
                str_kline += ',' + str(tup_kline[8])  # open interest
                str_kline += ',' + str(tup_kline[9])  # finance write 0
                str_kline += ',' + str(tup_kline[10])  # security lending volume write 0
                str_kline += '\n'
                f2.write(str_kline)
                pos += KLINE_SIZE
    finally:
        f1.close()
        f2.close()


def usage():
    print 'Binary2Csv.py usage:'
    print 'python Binary2Csv.py sh1|sh2|sho|shfi snap|tick|auction|kline binary_file_path csv_file_path'


def main(argv):
    if len(argv) != 5:
        print 'Error: please input right parameters'
        usage()
        sys.exit(2)
    market = argv[1]
    category = argv[2]
    infile = argv[3]
    outfile = argv[4]
    if market == 'sh1' and category == 'snap':
        convert_sh1_snap(infile, outfile)
    elif market == 'sho' and category == 'snap':
        convert_sho_snap(infile, outfile)
    elif market == 'sh2' and category == 'snap':
        convert_sh2_snap(infile, outfile)
    elif market == 'sh2' and category == 'tick':
        convert_sh2_tick(infile, outfile)
    elif market == 'sh2' and category == 'auction':
        convert_sh2_auction(infile, outfile)
    elif market == 'sh2' and category == 'kline':
        convert_sh1_sh2_kline(infile, outfile)


if __name__ == "__main__":
    main(sys.argv)
