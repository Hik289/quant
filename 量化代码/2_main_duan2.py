

import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
import cvxopt
# from optimization import convex_optimization as co
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
def convex_optimization(n, e_return, tc, x_style, x_industry,filter, F, delta, W_bm, W_last,W_min, te):
    # n company, e_return stands for expect return of alpha, x_style and x_industry is beta factor,F is factor covariance
    # matrix of factor([industry style]), delta is the covariance matrix of unique factor of each stock,W_bm is the
    # allocation of benchmark portfolio, W_last is the allocation of previous portfolio

    # X = [x_industry,x_style]
    X = np.hstack((x_industry, x_style))  # x_industry=n*29   x_style=m*3
    X = np.matrix(X)  # e_return=n*1

    X_tmp = np.matrix(X)
    X_trans = np.transpose(X_tmp)
    delta = np.matrix(delta)  # delta=n*n
    cov_matrix = np.dot(X, np.dot(F, X_trans)) + delta  # F=31*31
    W_bm = np.matrix(W_bm)  # W_bm=n*1
    W_last = np.matrix(W_last)  # W_last=n*1

    def Fo(x=None, z=None):

        if x is None:
            return 1, cvxopt.matrix(1.0 / n, (n, 1))
        f = np.matrix([0.0, 0.0]).T
        eb = 0.0025
        for i in range(0, n):
            tmp = float(x[i] - W_last[i])
            if abs(tmp) < eb:
                f[0] = f[0] + tc * (tmp ** 2) / (2 * eb) + (2 * eb) / 4
            elif tmp > 0:
                f[0] = f[0] + tc * (tmp)
            elif tmp < 0:
                f[0] = f[0] + tc * (-tmp)
        f[0] = f[0] - np.dot(x.T, e_return)
        f = cvxopt.matrix(f)
        De = cvxopt.matrix(x - W_bm)
        De_trans = De.T
        a = np.dot(De_trans, cov_matrix)
        b = np.dot(a, De)
        f[1] = float(b - te * te * 1.0 / 12)

        df = np.ones([2, n])
        for i in range(0, n):
            tmp = float(x[i] - W_last[i])
            if abs(tmp) < eb:
                df[0, i] = tc * tmp / eb - e_return[i]
            elif tmp > 0:
                df[0, i] = tc - e_return[i]
            elif tmp < 0:
                df[0, i] = -tc - e_return[i]
        df[1, :] = np.transpose(2 * np.dot(cov_matrix, (x - W_bm)))
        df = cvxopt.matrix(df)
        if (z == None):
            return f, df

        tmp = np.ones([1, n])
        for i in range(0, n):
            t = float(x[i] - W_last[i])
            if abs(t) < eb:
                tmp[0, i] = tc / eb
            elif abs(t) == eb:
                tmp[0, i] = 0.5 * tc / eb
            else:
                tmp[0, i] = 0

        h1 = cvxopt.spdiag(cvxopt.matrix(tmp))
        h2 = cvxopt.matrix(2 * cov_matrix)
        H = z[0] * h1 + z[1] * h2
        return f, df, H

    x1 = np.array(X_trans)
    N = X[:, 1].size  # M number of factor, N number of firm
    one1 = np.ones(N)

    b1 = np.dot(X_trans, W_bm)
    b2 = np.zeros((np.shape(filter)[1], 1))
    one2 = np.ones(1)

    if x_industry.sum(axis=1).sum() == n:
        x2 = np.row_stack((x1,filter.T))
        b1 = np.row_stack((b1, b2))
    else:
        x2 = np.row_stack((x1, filter.T,one1))
        b1 = np.row_stack((b1, b2, one2))


    G0 = -np.eye(n)
    G1=np.eye(n)
    G=np.row_stack((G0,G1))

    h0 = np.transpose(np.matrix((np.zeros(n))))
    h1=W_min
    h=np.row_stack((h0,h1))

    G = cvxopt.matrix(G)
    x2 = cvxopt.matrix(x2)
    b1 = cvxopt.matrix(b1)
    h = cvxopt.matrix(h)

    return cvxopt.solvers.cp(Fo, G=G, h=h, A=x2, b=b1,kktsolver='ldl',options = {'kktreg':1e-6})['x']



RESULT=pd.read_csv(dir_path+'/result.csv',index_col='S_INFO_WINDCODE')
result=np.nan_to_num(np.array(RESULT))
D=pd.read_csv(dir_path+'/Dmatrix.csv',index_col=[0])
D=np.nan_to_num(D)
F=pd.read_csv(dir_path+'/Fmatrix.csv',index_col=[0])
F=np.nan_to_num(F)
n=np.shape(result)[0]
num_count = n
num_industry = 28
e_return=result[:,3]
tc = 0.003
X_style=result[:,0:3]
X_industry=result[:,5:33]
F=F
delta=D
W_bm=np.reshape(result[:,4],(-1,1))/np.sum(result[:,4])

W_last=pd.read_csv(dir_path+'/WT_20160106.csv',index_col='S_INFO_WINDCODE')
W_last=np.array(W_last)
# W_last=np.reshape(np.zeros(n),(-1,1))
###########  得到对个股权重的限定 W_min ###############
W_in_bm=np.dot(X_industry.T,W_bm)
W_industry=np.dot(X_industry,W_in_bm)
C=np.hstack((np.reshape(0.05*np.ones(len(W_industry)),(-1,1)),W_industry))
W_min=np.zeros(len(C))
for i in range(len(C)):
    W_min[i]=min(C[i])
W_min=np.reshape(W_min,(-1,1))

############## filter #################
filter=pd.read_csv(dir_path+'/filter.csv',index_col=[0])

###########################################
# W_last=np.transpose(np.matrix(np.ones(n)/n))
te = 1
re = convex_optimization(n, e_return, tc, X_style, X_industry,filter, F, delta, W_bm, W_last,W_min, te)



A=pd.DataFrame(np.array(re),index=RESULT.index)
A.to_csv(dir_path+'/WT_20160107.csv')

# print(re)





