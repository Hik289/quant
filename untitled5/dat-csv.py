
import os
from handleHistoryData import *


def dat_to_csv(path_from, path_des, files):
    dir_path = os.path.join(path_from, files)

    file_name = os.path.splitext(files)[0]
    file_type = os.path.splitext(files)[1]

    file_test = open(dir_path, encoding='gb18030',errors="ignore")

    new_dir = os.path.join(path_des, str(file_name) + '.csv')
    print(new_dir)

    file_test2 = open(new_dir, 'w')
    a=1

    for lines in file_test.readlines():
        print(a)
        str_data = ",".join(lines.split('\t'))
        file_test2.write(str_data)
        a+=1
    file_test.close()
    file_test2.close()

if __name__ == '__main__':
    marketVer = ['sh2', 'sz2', 'sho']
    dataDir = '/mnt/hisdata/binary'
    outDir = 'home/mds/HisTick'
    dateRange = getTradingDayRange(20180101,20180531)
    for Version in marketVer:
        for d in dateRange.astype(str):
            print('transferring '+Version+' '+d+' data')
            path_from= dataDir+'/'+Version+'/'+d
            path_des = outDir + '/' + Version + '/' + d
            print('transferring ' + Version + ' ' + d + ' data to'+ path_des)

            for files in os.listdir(path_from):
                if files== 'HisTick.dat':
                    if os.path.exists(path_des) == False:
                        os.makedirs(path_des)
                    dat_to_csv(path_from, path_des, files)
                    print('OK')

        #data = xr.concat([handle_minuteBar(pd.read_csv(os.path.join(dataDir, dataVer, m, d, 'Minute.csv'), skiprows=1, header=None), m[:2].upper()) for m in marketVer], dim='ticker')
        #data.to_netcdf(os.path.join(outDir, d+'.h5'))
