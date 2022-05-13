# _*_coding:utf-8_*_
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['font.sans-serif']=['SimHei']
tra_accuracy=[0.12,0.23,0.31,0.34,0.43,0.51,0.55,0.66,0.68,0.74,0.8,0.9]
test_acc = [0.4,0.5,0.6,0.7,0.8,0.9]

a=np.linspace(0,100,2)
print(a)
#正确率绘图
fig1=plt.figure('fig1')
plt.plot(np.linspace(0, 11, len(tra_accuracy)),tra_accuracy,'b-',label='训练的正确率')
plt.plot(np.linspace(0, 10, len(test_acc)),test_acc,'k-.',label='测试的正确率')
plt.title('训练、测试的正确率')
plt.xlabel('迭代次数')
plt.ylabel('准确率')
plt.legend(loc='lower right')

plt.grid()
plt.show()
