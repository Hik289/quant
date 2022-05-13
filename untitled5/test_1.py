import os

path_from= 'C:\Users\ThinkPad'
path_des='C:\Users\ThinkPad'

def dat_to_csv(path_from, path_des, files):
    dir_path = os.path.join(path_from, files)

    file_name = os.path.splitext(files)[0]
    file_type = os.path.splitext(files)[1]

    file_test = open(dir_path)

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
    dat_to_csv(path_from,path_des,'HisSnap.dat')