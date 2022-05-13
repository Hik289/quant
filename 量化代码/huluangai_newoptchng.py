import numpy as np
import pandas as pd
import os
#import cvxopt
from collections import Counter
import matplotlib.pyplot as plt
import base_class
import time
import config
# import mosek
import cvxpy
#import sys
from datetime import datetime,timedelta
opttime=time.time()
class opt():
    def __init__(self, start, end, risk_factor_list, residual_list, alpha_sign, coef_all_df,residual_df):
        self.infeasible_day = []
        self.out_of_bound_weight_day=[]
        self.alpha_sign = alpha_sign
        self.bb = base_class.base_class(risk_factor_list)
        self.risk_factor_list = self.bb.read_risk_factors()
        self.risk_number = len(risk_factor_list)
        self.timelist = self.bb.timelist
        while start not in self.timelist:
            start=(pd.to_datetime(start)+timedelta(1)).strftime('%Y%m%d')
        while end not in self.timelist:
            end=(pd.to_datetime(end)+timedelta(-1)).strftime('%Y%m%d')
        self.Return = self.bb.other_factors[0]
        self.industry = self.bb.other_factors[1]
        self.industry_number = len(self.industry.columns)
        #self.weight = self.bb.other_factors[2]
        self.Return_after = self.Return.shift(periods=-1, axis=1)

        #
        #self.expcoefdat = []
        #self.expcoefadjdat = []
        self.start_date_time = start
        self.end_date_time = end
        self.residual = residual_df
        self.residual_list = residual_list
        self.alpha_number = len(residual_list)
        self.coef = coef_all_df.iloc[:self.alpha_number,:]
        self.fff = pd.concat([coef_all_df.iloc[-self.industry_number:, :],
                              coef_all_df.iloc[self.alpha_number:(self.alpha_number + self.risk_number), :],
                              coef_all_df.iloc[:self.alpha_number, :]], axis=0)
        self.start = self.timelist.index(start)
        self.end = self.timelist.index(end)
        self.WEIGHT = pd.DataFrame(index=self.Return.index)
        
        #
        #self.ansadjpos = []
        self.opt_process()
        self.plotdata()
        self.print_result()

    def get_F(f_factors, day, h, lamda):  # 针对表里第t行对应的日子,记得加单引号
        def standard_aa(array):
            return array - array.mean(0)

        Day = f_factors.columns.size
        K = len(f_factors.index)
        index_1 = np.arange(h + 1)
        index_1 = index_1[::-1]
        lamda_aa = np.ones(h + 1)
        lamda_aa = (lamda_aa * lamda) ** index_1
        t = f_factors.columns.values.tolist().index(day)
        F = np.zeros((K, K))
        b = f_factors.iloc[:, t - h:t + 1].values.T
        for k in range(0, K):
            for k_ in range(0, k + 1):
                f_k = standard_aa(b[:, k])
                f_k_ = standard_aa(b[:, k_])

                F[k, k_] = (f_k * f_k_ * lamda_aa).sum() / lamda_aa.sum()
                F[k_, k] = F[k, k_]
        return F

    def get_Delta(residual, day, h, lamda):
        def standard_aa(array):
            return array - array.mean(0)

        u_residual = residual.copy()
        N = u_residual.iloc[:, 0].size
        D = np.zeros((N, N))
        index_1 = np.arange(1, h + 2)
        index_1 = index_1[::-1]
        lamda_aa = np.ones(h + 1)
        lamda_aa = (lamda_aa * lamda) ** index_1
        t = u_residual.columns.values.tolist().index(day)
        b = u_residual.iloc[:, t - h:t + 1].values.T

        for k in range(0, N):
            f_k = b[:, k]
            lamda_tem = lamda_aa.copy()
            Nu_Nums = Counter(np.isnan(f_k))[1]
            if Nu_Nums == 0:
                f_k = standard_aa(f_k)
                D[k, k] = (f_k * f_k * lamda_tem).sum() / lamda_tem.sum()
            elif Nu_Nums <= 0.2 * h:
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

    def get_expect_return(self,day, coef, h, data_real, F, D_real,WEIGHT_min_va, IR_year,itime):  
        # 过去h天coef的半衰加权作为expect_return
        if np.isreal(h) == True and h!=0:
            index_1 = np.arange(h + 1)
            index_1 = index_1[::-1]
            lamda_aa = np.ones(h + 1)
            lamda = (0.5) ** (1 / 15)
            lamda_aa = (lamda_aa * lamda) ** index_1
            t = coef.columns.values.tolist().index(day)
            b = coef.iloc[:, t - h:t + 1].values.T
            n = len(coef.index) #41+(# of alpha)
            ans = np.zeros((n, 1))
            for i in range(0, n):
                ans[i] = (b[:, i] * lamda_aa).sum() / lamda_aa.sum()
            ansadj = ans-np.array([coef.iloc[:, t - h:t].T.std()]).T
            
            #self.expcoefdat.append(ans[n-1][0])
            #self.expcoefadjdat.append(ansadj[n-1][0])
            #self.ansadjpos.append((ansadj>0).sum())
            return ansadj

        if h == 'mv':
            expcoef = []
            X2 = np.mat(data_real.values)#2818,42
            Fmat = np.mat(F)#42，42
            DD = np.mat(D_real.values)#2818,2818
            sig = X2 * Fmat * X2.T + DD#2818,2818
            for j in range(0, self.alpha_number):
                w = WEIGHT_min_va[j].ix[:, itime - self.start]
                w = w.dropna()
                w = np.mat(w.values)
                tem1 = np.sqrt((w * sig * w.T)[0, 0])  
                expcoef.append(pow((1 + IR_year[j] * tem1 * np.sqrt(250)), 1 / 250) - 1 - config.opt_object_lambda * tem1 * tem1)
                #expcoef.append(pow((1 + IR_year[j] * tem1 * np.sqrt(250)), 1 / 250) - 1)
            ans = np.array([expcoef]).T
            return ans

    def min_variance_opimization(F, D, X, A):
        # sigma1 = np.mat(sigma.values)
        # sigma1_inverse = sigma1.I
        stocklist = X.index
        N, M = X.shape
        one1 = np.ones(N)
        X1 = np.column_stack((X.values, one1))
        X1 = np.mat(X1)
        A1 = A.values
        one2 = np.ones(1)
        A1 = np.row_stack((A1, one2))
        A1 = np.mat(A1)
        # A1 = A1[32:,:]
        D = np.mat(D)
        F = np.mat(F)
        X2 = np.mat(X.values)
        sigma1 = X2 * F * X2.T + D
        sigma1_inverse = sigma1.I
        X1 = X.values
        X1 = np.mat(X1)
        A1 = A.values
        A1 = np.mat(A1)
        # w = A1*(X1.T*sigma1_inverse*X1).I*sigma1_inverse*X1
        w = sigma1_inverse * X1 * np.linalg.pinv(X1.T * sigma1_inverse * X1) * A1
        weight = pd.DataFrame(data=w, index=stocklist)
        return weight

    def cvx_op(times,cvxpy_solver, N, e_return, tc, te, W_bm_real, W_last, x_industry, x_style, data_real, D_real, F,
               factor_bm, method, inaccurate_day=0):

        e_return_np = np.array(e_return)
        F_np = np.array(F)
        D_real_np = np.array(D_real)
        factor_bm_np = np.array(factor_bm)
        W_bm_real_np = np.array(W_bm_real)
        W_last_np = np.array(W_last)
        x_style_industry_np = np.array(pd.concat([x_industry, x_style], axis=1))
        X_np = np.array(data_real)
        te = te * te / 252

        w = cvxpy.Variable((N, 1))
        f = X_np.T * (w - W_bm_real_np)
        temax = cvxpy.Parameter()
        temax.value = te

        ret = e_return_np.T * w
        if method == 'abs':
            cost = 0.5 * tc * cvxpy.sum(cvxpy.abs(w - W_last_np))
        elif method == 4 / 3:
            cost = 0.5 * tc * cvxpy.sum((w - W_last_np) ** (4 / 3))
        elif method == 2:
            cost = 0.5 * tc * cvxpy.sum((w - W_last_np) ** 2 / 0.005 + 0.00125)
        risk = cvxpy.quad_form(f, F) + cvxpy.quad_form(w - W_bm_real_np, D_real_np)
        mid = x_style_industry_np.T * w
        if config.opt_object_function == 0:
            object_function = ret - cost
        else:
            object_function = ret - config.opt_object_lambda * risk - cost
        if config.opt_constraint == 1:
            prob_factor = cvxpy.Problem(cvxpy.Maximize(object_function),
                                        [cvxpy.sum(w) == 1, w >= 0,
                                         mid == factor_bm_np,
                                         risk*times <= temax*times
                                         ])
        else:
            prob_factor = cvxpy.Problem(cvxpy.Maximize(object_function),
                                        [cvxpy.sum(w) == 1, w >= 0,
                                         mid == factor_bm_np,
                                         ])
        prob_factor.solve(solver=cvxpy_solver, verbose=True)  # solver=MOSEK,
        if prob_factor.status == 'infeasible':
            return np.nan
        else:
            return w.value

    def cvx_op_open(times,cvxpy_solver, N, e_return, tc, te, W_bm_real, W_last, x_industry, x_style, data_real, D_real, F,
                    factor_bm, method, inaccurate_day=0):
#数据准备
        #factor_bm表示两个中性约束（即将组合与bm对比bias为0）
        e_return_np = np.array(e_return)
        F_np = np.array(F)
        D_real_np = np.array(D_real)
        factor_bm_np = np.array(factor_bm)
        W_bm_real_np = np.array(W_bm_real)
        W_last_np = np.array(W_last)
        X_np = np.array(data_real)
        x_industry_np = np.array(x_industry)
        x_style_np = np.array(x_style)
        te = te * te / 252
        #将te平方，并转成daily的tracking error
        w = cvxpy.Variable((N, 1))
        f = X_np.T * (w - W_bm_real_np)
        temax = cvxpy.Parameter()
        temax.value = te
#期望组合收益
        ret = e_return_np.T * w
#计算惩罚函数
        if method == 'abs':
            cost = 0.5 * tc * cvxpy.sum(cvxpy.abs(w - W_last_np))
        elif method == 4 / 3:
            cost = 0.5 * tc * cvxpy.sum((w - W_last_np) ** (4 / 3))
        elif method == 2:
            cost = 0.5 * tc * cvxpy.sum((w - W_last_np) ** 2 / 0.005 + 0.00125)

        factor_up = np.vstack(
            np.hstack(factor_bm_np[32:41]) + np.array(config.opt_risk_upperbound))
        factor_low = np.vstack(
            np.hstack(factor_bm_np[32:41]) + np.array(config.opt_risk_lowerbound))
#计算约束条件
        #.quad_form是二次型
        risk = cvxpy.quad_form(f, F) + cvxpy.quad_form(w - W_bm_real_np, D_real_np)
        industry_mid = x_industry_np.T * w
        style_mid = x_style_np.T * w
#确定目标函数
        if config.opt_object_function == 0:
            object_function = ret - cost
        else:
            object_function = ret - config.opt_object_lambda * risk - cost
#是否添加二阶约束条件（σ risk），定义优化的问题和约束条件
        if config.opt_constraint == 1:
            prob_factor = cvxpy.Problem(cvxpy.Maximize(object_function),
                                        [cvxpy.sum(w) == 1, w >= 0,
                                         industry_mid <= (1+config.opt_industry_upperbound) * factor_bm_np[0:32],
                                         industry_mid >= (1+config.opt_industry_lowerbound) * factor_bm_np[0:32],
                                         style_mid <= factor_up,
                                         style_mid >= factor_low,
                                         risk*times <= temax*times
                                         ])
        else:
            prob_factor = cvxpy.Problem(cvxpy.Maximize(object_function),
                                        [cvxpy.sum(w) == 1, w >= 0,
                                         industry_mid <= (1+config.opt_industry_upperbound) * factor_bm_np[0:32],
                                         industry_mid >= (1+config.opt_industry_lowerbound) * factor_bm_np[0:32],
                                         style_mid <= factor_up,
                                         style_mid >= factor_low,
                                         ])
#执行优化过程
        prob_factor.solve(solver=cvxpy_solver, verbose=True, max_iters=5000)
        if prob_factor.status == 'infeasible':
            return np.nan
        else:
            return w.value

    def opt_process(self):
        date_available = 0 #-1
        WEIGHT_min_va = []
        IR_year = []
        
        # loop for optimization
        turnover = []        
        for i in range(self.start, self.end):
            print(self.timelist[i])
            dat = self.bb.getDataReady(self.timelist[i-config.predict_period+1 + date_available ], 0,True, *self.residual_list)
            data = pd.concat([dat.iloc[:, -self.industry_number:],
                              dat.iloc[:, self.alpha_number:(self.alpha_number + self.risk_number)],
                              dat.iloc[:, :self.alpha_number]], axis=1)
            x_alpha = data.iloc[:, -self.alpha_number:]
            F = opt.get_F(self.fff, self.timelist[i-config.predict_period + date_available], config.opt_window, 0.5 ** (5 / config.opt_window))
            D = opt.get_Delta(self.residual, self.timelist[i-config.predict_period + date_available], config.opt_window, 0.5 ** (5 / config.opt_window))
            D1 = pd.DataFrame(data=D, index=self.residual.index, columns=self.residual.index)
            l = data.index
            D2 = pd.DataFrame(data=D1, index=l, columns=l)
            D3 = D2.dropna(how='any')
            real_list = D3.index  # data,D
            D_real = pd.DataFrame(data=D2, index=real_list, columns=real_list)
            data_real = pd.DataFrame(data=data, index=real_list)
            print('data_real',data_real.Industry_1.size)
            x_alpha = pd.DataFrame(data=x_alpha, index=real_list)
            N = len(data_real.index)
                       
            if config.opt_exp_method == 0:
                exp_coef = np.zeros((self.alpha_number + self.industry_number + self.risk_number, 1))
                exp_coef[:] = 0.001
            else:
                # exp_coef = opt.get_expect_return(self.timelist[i-config.predict_period-1], self.fff, config.opt_exp_method)
                #exp_coef = opt.get_expect_return(self.timelist[i-config.predict_period + date_available], self.fff, config.opt_exp_method)
                exp_coef = self.get_expect_return(self.timelist[i-config.predict_period+date_available], self.fff, config.opt_exp_method, data_real, F, D_real,WEIGHT_min_va, IR_year,i)
                        
            x_industry = data_real.iloc[:, :self.industry_number]
            x_style = data_real.iloc[:, self.industry_number:self.industry_number + self.risk_number]
            M = len(data_real.columns)
            a = np.zeros([N, 1])
            
            if np.isreal(config.opt_exp_method) == True:
                #for j in range(0, self.alpha_number + self.industry_number + self.risk_number):
                for j in range(self.industry_number + self.risk_number, self.alpha_number + self.industry_number + self.risk_number):
                    b = np.mat(data_real.ix[:, j].values).T * exp_coef[j]
                    a = a + b
            else:
                for j in range(0, self.alpha_number):
                    b = np.mat(x_alpha.ix[:, j].values).T * exp_coef[j]
                    a += b

            e_return = pd.DataFrame(data=a, index=x_style.index, columns=['return'])

            W_bm = config.W_BM[self.timelist[i + date_available - 1 ]]
            W_bm = W_bm / W_bm.sum()
            W_bm = W_bm.dropna()
            W_bm_real = pd.DataFrame(data=W_bm, index=data_real.index)
            X_bm = pd.DataFrame(data=data.iloc[:, :self.industry_number + self.risk_number], index=W_bm.index)
            X_bm = X_bm.dropna()
            W_bm = pd.DataFrame(data=W_bm, index=X_bm.index)
            W_bm = W_bm / W_bm.values.sum()
            w1 = np.mat(W_bm.values)
            x1 = np.mat(X_bm.values)
            factor_bm = x1.T * w1
            W_bm_real = W_bm_real.fillna(0)
            W_bm_real = W_bm_real / W_bm_real.sum()

            if i == self.start:
                W_last = np.zeros((N, 1))
            else:
                W_last = self.WEIGHT[self.timelist[i - 1]].copy()
                W_last_real = pd.DataFrame(data=W_last, index=data_real.index)
                W_last_real = W_last_real / W_last_real.sum()
                W_last_real = W_last_real.fillna(0)
                W_last = W_last_real

            if config.opt_pkg == 'cpy':
                if config.opt_exposure == 0:
                    ans1 = opt.cvx_op(config.opt_exaggerate_times,config.opt_cvxpy_solver, N, e_return, config.opt_tc, config.opt_te, W_bm_real,
                                      W_last, x_industry, x_style, data_real, D_real, F, factor_bm,
                                      config.opt_method)
                elif config.opt_exposure == 1:
                    ans1 = opt.cvx_op_open(config.opt_exaggerate_times,config.opt_cvxpy_solver, N, e_return, config.opt_tc, config.opt_te,
                                           W_bm_real, W_last, x_industry, x_style, data_real, D_real, F, factor_bm,
                                           config.opt_method)

            print('%sweight之和(before强行归一化):'%self.timelist[i],np.sum(ans1))
            if type(ans1) == float:
                self.infeasible_day.append(self.timelist[i])
                self.WEIGHT[self.timelist[i]] = W_last
            elif (np.sum(ans1)>config.weight_upperbound)|(np.sum(ans1)<config.weight_lowerbound):
                self.out_of_bound_weight_day.append(self.timelist[i])
                self.WEIGHT[self.timelist[i]] = W_last
            else:
                ans1=ans1/np.sum(ans1)#######强行归一化
                self.WEIGHT[self.timelist[i]] = pd.DataFrame(data=ans1, index=data_real.index)
            # 计算换手率：
            if i == self.start:
                turnover.append(np.nan)
            else:
                turnover.append(np.sum(np.abs(self.WEIGHT[self.timelist[i]] - self.WEIGHT[self.timelist[i - 1]])))
        
        
        # 跳出t的循环,还在函数里
        TurnOver = pd.DataFrame(data=turnover, index=self.timelist[self.start:self.end])
        self.TurnOverRate = TurnOver.mean()

    def plotdata(self):
        plt.close('all')
        ret_after = self.Return.shift(periods=-1, axis=1).iloc[:, self.start:self.end]
        R = (ret_after * self.WEIGHT).sum()
        # R.values[0] = R.values[0] - config.opt_tc * np.sum(np.abs(self.WEIGHT[self.timelist[self.start]])) / 2

        if config.opt_method == 'abs':
        # if 'abs'=='abs':
            R.values[0] = R.values[0] - config.opt_tc * np.sum(np.abs(self.WEIGHT[self.timelist[self.start]])) / 2
            for i in range(self.start + 1, self.end):
                R.values[i - self.start] = R.values[i - self.start] - config.opt_tc * np.sum(
                    np.abs((self.WEIGHT[self.timelist[i]] - self.WEIGHT[self.timelist[i - 1]]))) / 2
        elif config.opt_method == 4 / 3:
            R.values[0] = R.values[0] - config.opt_tc * np.sum((self.WEIGHT[self.timelist[self.start]]) ** 4 / 3) / 2
            for i in range(self.start + 1, self.end):
                R.values[i - self.start] = R.values[i - self.start] - config.opt_tc * np.sum(
                    np.abs((self.WEIGHT[self.timelist[i]] - self.WEIGHT[self.timelist[i - 1]]) ** 4 / 3)) / 2
        elif config.opt_method == 2:
            R.values[0] = R.values[0] - config.opt_tc * np.sum(
                (self.WEIGHT[self.timelist[self.start]]) ** 2 / 0.005 + 0.00125) / 2
            for i in range(self.start + 1, self.end):
                R.values[i - self.start] = R.values[i - self.start] - config.opt_tc * np.sum(
                    (self.WEIGHT[self.timelist[i]] - self.WEIGHT[self.timelist[i - 1]]) ** 2 / 0.005 + 0.00125) / 2
        # for i in range(self.start + 1, self.end):
        #     R.values[i - self.start] = R.values[i - self.start] - config.opt_tc * np.sum(np.abs((self.WEIGHT[self.timelist[i]] - self.WEIGHT[self.timelist[i - 1]]))) / 2

        W_bm_temp = config.W_BM.ix[:, self.timelist[self.start]:self.timelist[self.end - 1]] / 100
        R_bm = (ret_after * W_bm_temp).sum()
        R_real = R - R_bm
        RR = (R_real + 1).cumprod()
        RR.index = pd.to_datetime(RR.index)
        RR.plot()
        plt.show()
        plt.close()
        self.tracking_error = np.sqrt(252) * R_real.std()
        self.sharpe_ratio = (RR[-1]**(252/len(RR))-1)/self.tracking_error
        count=len(RR)
        max_DD_list = np.zeros(count)
        for i in range(1,count-1):
            max_DD_list[i] = (RR[i] - RR[i:count-1].min())/RR[i]
        self.max_DD = max(max_DD_list)
        self.annreturn = RR[-1]**(252/len(RR))-1
        stats_result=pd.DataFrame([self.TurnOverRate[0],self.tracking_error,self.sharpe_ratio,self.max_DD,self.annreturn,config.opt_object_function == 1,config.opt_constraint == 1,config.opt_exp_method,config.opt_method,config.opt_pkg,config.opt_te,config.opt_tc,config.opt_object_lambda,len(self.infeasible_day),len(self.out_of_bound_weight_day)],index=['换手率','跟踪误差(年化波动率)','夏普比率','最大回撤','年化收益率','目标函数是否有风险','限制条件是否有风险','exp_method','目标函数成本函数形式','优化包','te','tc','lambda','不可解天数','WeightOutofBound天数'])
        writer = pd.ExcelWriter(config.opt_des_path+'/obj%s_constrain%s_lambda%s_opt%s_te%s_tc%s_%s_%s_t%s.xlsx'%(config.opt_object_function,config.opt_constraint,config.opt_object_lambda,config.opt_pkg,config.opt_te,config.opt_tc,self.start_date_time[2:],self.end_date_time[2:],config.predict_period))
        RR.to_excel(writer,'Data4NetValueCurve')
        stats_result.to_excel(writer,'stats_result')
        self.WEIGHT.to_excel(writer,'WEIGHT')
        if len(self.infeasible_day)>0:
            indf=pd.DataFrame(self.infeasible_day)
            indf.to_excel(writer,'infeasible_day')
        if len(self.out_of_bound_weight_day)>0:
            oudf=pd.DataFrame(self.out_of_bound_weight_day)
            oudf.to_excel(writer,'out_of_bound_weight_day')
        writer.close()
        
    def print_result(self):
        print('time used for total optimization:',time.time()-opttime)
        print('换手率:', self.TurnOverRate[0])
        print('跟踪误差(年化波动率):', self.tracking_error)
        print('夏普比率:', self.sharpe_ratio)
        print('最大回撤:', self.max_DD)
        print('年化收益率:', self.annreturn)
        print('目标函数是否有风险:',config.opt_object_function == 1)
        print('限制条件是否有风险:',config.opt_constraint == 1)
        print('exp_method:',config.opt_exp_method)
        print('目标函数成本函数形式:',config.opt_method)
        print('优化包:',config.opt_pkg)
        print('te:',config.opt_te)
        print('tc:',config.opt_tc)
        print('lambda:',config.opt_object_lambda)
        print('不可解天数:',len(self.infeasible_day))
        print('WeightOutofBound天数:',len(self.out_of_bound_weight_day))