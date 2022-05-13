from collections import Counter
import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
import cvxopt
# from optimization import convex_optimization as co
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

def convex_optimization(e_return, tc, x_style, x_industry,filter, F, delta, W_bm, W_last,W_min, te):
    # n company, e_return stands for expect return of alpha, x_style and x_industry is beta factor,F is factor covariance
    # matrix of factor([industry style]), delta is the covariance matrix of unique factor of each stock,W_bm is the
    # allocation of benchmark portfolio, W_last is the allocation of previous portfolio
    n=np.shape(e_return)[0]

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
    one1 = np.ones(n)

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

    # return cvxopt.solvers.cp(Fo, G=G, h=h, A=x2, b=b1,kktsolver='ldl',options = {'kktreg':1e-6,'maxiters':5,'feastol':1e-3})['x']
    result=cvxopt.solvers.cp(Fo, G=G, h=h, A=x2, b=b1,kktsolver='ldl',options = {'kktreg':1e-6,'maxiters':25,'feastol':1e-3})
    return result

#######################   Month   ##########################
#########################    F and delta   ###############################################
lamda=0.5**(1/60)
def get_F(factors,day,h,lamda):#针对表里第t行对应的日子,记得加单引号
    def standard_aa(array):
        return array-array.mean(0)

    f_factors=factors.copy()
    Day=f_factors.columns.size
    K=f_factors.iloc[:,0].size
    index_1=np.arange(h+1)
    index_1 = index_1[::-1]
    lamda_aa=np.ones(h+1)
    lamda_aa=(lamda_aa*lamda)**index_1
    t=f_factors.columns.values.tolist().index(day)
    F=np.zeros((K,K))
    b=f_factors.iloc[:,t-h:t+1].values.T
    #sigma_t=sigma.iloc[:,t-h:t+1].values.T
    #B_2=((b/sigma_t)**2).mean(1)
    #lamda_2=(B_2*lamda_aa).sum()
    for k in range(0,K):
        for k_ in range(k+1):
            f_k=standard_aa(b[:,k])
            f_k_=standard_aa(b[:,k_])
            F[k,k_]=(f_k*f_k_*lamda_aa).sum()/lamda_aa.sum()
            F[k_,k]=F[k,k_]
    #F=F*lamda_2
    return F

def get_Delta(residual, day, h, lamda):
    def standard_aa(array):
        return array - array.mean(0)
    u_residual=residual.copy()
    N = u_residual.iloc[:, 0].size
    D = np.zeros((N, N))
    index_1 = np.arange(1, h + 2)
    index_1 = index_1[::-1]
    lamda_aa = np.ones(h + 1)
    lamda_aa = (lamda_aa * lamda) ** index_1
    t = u_residual.columns.values.tolist().index(day)
    b = u_residual.iloc[:, t - h:t + 1].values.T
    for k in range(0, N):
        # print ("k is %d "%k)
        f_k = b[:, k]
        lamda_tem = lamda_aa.copy()
        Nu_Nums = Counter(np.isnan(f_k))[1]
        # print("num is %d"%Nu_Nums)
        if Nu_Nums == 0:
            f_k = standard_aa(f_k)
            D[k, k] = (f_k * f_k * lamda_tem).sum() / lamda_tem.sum()
        elif Nu_Nums <= 0.5 * h:
            temp = np.argwhere(np.isnan(f_k)).tolist()
            for index in range(0, Nu_Nums):
                x1 = temp[index][0]  # 第index个NaN所在坐标位置
                lamda_tem[x1] = 0
                f_k[x1] = 0

            f_k = standard_aa(f_k)
            D[k, k] = (f_k * f_k * lamda_tem).sum() / lamda_tem.sum()  # 原位置上的权重不变
        else:
            D[k, k] = np.nan
    return D


#######################数据读入 #########################
# 数据的读入
# dir_path=r'E:\SVN\real_cp'
ALPHA_RETURN=pd.read_csv(dir_path+r'/alpha_return1.csv',index_col=[0])

BETA=pd.read_csv(dir_path +r'/beta_own_adj.csv',index_col='S_INFO_WINDCODE')
SIZE=pd.read_csv(dir_path +'/circulation_value_ln1adj2.csv',index_col='S_INFO_WINDCODE')
VALUE=pd.read_csv(dir_path +r'/ln_pb.csv',index_col='S_INFO_WINDCODE')
industry=pd.read_csv(dir_path +r'/industry_new.csv',index_col='S_INFO_WINDCODE')


# industry=INDUSTRY.drop(columns=['Industry_1','Industry_5','Industry_13'])
RET=pd.read_csv(dir_path +'/daily_return.csv',index_col='S_INFO_WINDCODE')
W_BM=pd.read_csv(dir_path +'/W_bm_hs_300.csv',index_col=[0]) .ix[:,388:]   #300*days  # tushare   387 是2009年8月3日
residual=pd.read_csv(dir_path +r'/alpha_return1.csv',index_col=[0])
WT=pd.DataFrame(np.zeros((np.shape(residual)[0],np.shape(W_BM)[1])),index=residual.index,columns=W_BM.columns)
fff=pd.read_csv(dir_path +r'/coef1.csv',index_col=[0])
Change=pd.DataFrame(np.zeros((9,np.shape(W_BM)[1])),index=['Change','Return','HS300','alpha','risk','sigma','alpha_return','residual','find'],columns=W_BM.columns)
day_code=W_BM.columns
for i in range(1,200):
    day0=day_code[i-1]
    day=day_code[i]
    print(day)
    print(i)

    ################# F & D #########################
    F=np.nan_to_num(get_F(fff,day,250,lamda))#针对表里第t行对应的日子,记得加单引号
    # dF = pd.DataFrame(FF ,columns=fff.index.values.tolist(), index=fff.index.values.tolist())      #######   dF   #####
    # F=np.nan_to_num(dF)

    delta=np.nan_to_num(get_Delta(residual,day,250,lamda))
    # dD = pd.DataFrame(DD ,columns=residual.index.values.tolist(), index=residual.index.values.tolist())
    # delta=np.nan_to_num(dD)

    ############## filter #################
    code=residual[day]
    ###############  filter 矩阵为了确定股票池 非今天的股票位置=1 ################
    num_all=np.shape(code)[0]
    code_drop=code.drop(code.index[np.where(~np.isnan(code))])
    num_drop=np.shape(code_drop)[0]

    filter=pd.DataFrame(np.zeros((num_all,num_drop)),index=code.index,columns=code_drop.index)
    for i in code_drop.index:
        filter.loc[i, i] = 1
    filter=np.array(filter)
    f=fff.ix[:,day0]
    size=SIZE[day0]
    value=VALUE[day0]
    beta=BETA[day0]
    ret=RET[day]
    W_bm=W_BM[day0]
    W_last=WT[day0]
    alpha_return=ALPHA_RETURN[day0]
    result = pd.concat([size,value,beta,ret,W_bm,industry,code,W_last,alpha_return], axis=1, join_axes=[code.index])
    result=np.nan_to_num(np.array(result))

    X_style=result[:,0:3]                                                              #  n*3
    e_return=result[:,3]                                                                    #  n*1
    W_bm=np.reshape(result[:,4],(-1,1))/np.sum(result[:,4])
    X_industry=result[:,5:37]                                                            #  n*28
    W_last=np.reshape(result[:,38],(-1,1))
    alpha_return=np.reshape(result[:,39],(-1,1))
    tc=0.003
    te=1

    # W_last=np.reshape(np.zeros(n),(-1,1))
    ###########  得到对个股权重的限定 W_min ###############
    W_in_bm=np.dot(X_industry.T,W_bm)
    W_industry=np.dot(X_industry,W_in_bm)
    C=np.hstack((np.reshape(0.05*np.ones(len(W_industry)),(-1,1)),W_industry))
    W_min=np.zeros(len(C))
    for i in range(len(C)):
        W_min[i]=min(C[i])
    W_min=np.reshape(W_min,(-1,1))

    # ############## filter #################
    # filter=pd.read_csv(dir_path+'/filter.csv',index_col=[0])

    ###########################################
    # W_last=np.transpose(np.matrix(np.ones(n)/n))

    RESULT = convex_optimization(alpha_return, tc, X_style, X_industry,filter, F, delta, W_bm, W_last,W_min, te)
    re=RESULT['x']
    find=(RESULT['status']=='optimal')
    if find==True:
        find=1
    elif find==False:
        find=0
    WT[day] = np.array(re)
    Change.loc['Change', day] = np.sum(np.abs(re - W_last))
    Change.loc['Return', day] = np.dot(re.T, e_return)
    Change.loc['HS300', day] = np.dot(W_bm.T, e_return)
    Change.loc['alpha', day] = Change.loc['Return', day]-Change.loc['HS300', day]
    # Change.loc['alpha_return',day]=np.dot(re.T,alpha_return)
    A=np.dot(np.hstack((X_industry,X_style)),f.T)
    Change.loc['residual',day]=np.mean(e_return-np.reshape(alpha_return,(1,-1))-A)
    # Change.loc['risk', day] = np.dot((re - W_bm).T, X_style[:, 2])  # size的风险敞口

    Change.loc['find',day]=find
    X = np.hstack((X_industry, X_style))
    cov_matrix = np.dot(X, np.dot(F, X.T)) + delta

    # Change.loc['sigma', day] = np.sqrt(np.dot((re - W_bm).T, np.dot(cov_matrix, (re - W_bm))))

Change.loc['net', :] = np.cumprod(Change.loc['alpha', :] + 1)

WT.to_csv(dir_path + r'/result/500_day_WT.csv')
Change.to_csv(dir_path + r'/result/500_day_change.csv')
print(Change)
# print(Return)