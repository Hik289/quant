#import multifactor
import pick_alpha
#import newopt
import pandas as pd
import config

start_date_time = '20110106'
end_date_time = '20180831'
'''
alpha_list=[1,5,6,11,38,47,48,60,81,91]
alpha_sign=[1,5,6,-11, 38, 47, 48, -60,-81,91]
alpha_list=[20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]
# alpha_list=[1, 5, 6, 7, 11, 12, 13, 16]
# alpha_sign=[1, 5, 6, -7, -11, 12, -13, 16]
alpha_list=[1, 2, 5, 6, 7, 11, 12, 16]
alpha_sign=[1, 2, 5, 6, -7, -11, 12, 16]
# alpha_list=[1, 2, 5, 6, 7, 8, 10, 11, 12, 16, 26, 32, 36, 39, 45, 48]
# alpha_sign=[1, -2, 5, 6,-7, 8, -10, -11, 12, 16, 26, 32, -36, -39, -45, 48]
alpha_list=[1, 2, 5, 10, 11, 12, 26, 32, 36, 39, 45, 48, 55, 60, 62, 64, 65]
alpha_list=[62, 80, 99, 26, 163, 184, 36, 45, 55, 64, 91, 100, 137, 11, 156, 176, 1, 111, 12, 139, 168, 2, 39, 48, 179, 189, 5, 114, 32, 60, 142, 160, 777]
alpha_list=[62, 80, 99, 26, 163, 184, 36, 45, 55, 64, 91, 100, 137, 11, 156, 176, 1, 111, 12, 139, 168, 2, 39, 48, 179, 189, 5, 114, 32, 60, 142, 777]
# alpha_list=[55, 64, 91, 100, 137, 11, 156, 176, 1, 111, 12]
# alpha_list=[139, 168, 2, 39, 48, 179, 189, 5, 32, 60, 142, 160, 777]
# alpha_list=[114]
# alpha_sign=[1, -2, 5,-7, -10, -11, 12, 26, 32, -36, -39, -45, 48, -55, -60, 62, -64, 65]
# alpha_list=[26, 32, 36, 39, 45, 48]
# alpha_sign=[26, 32, -36, -39, -45, 48]
# alpha_list=[55, 60, 62, 64, 65]
# alpha_sign=[-55, -60, 62, -64, 65]
alpha_list=[137]
# alpha_sign=[1,5,-7,-11,32,36]
'''
#alpha_list=[83,84,85,86,88,94,95,96,98,99]
alpha_list=list(range(101,111))
strongest_alpha_number = alpha_list
risk_factor_list = ['beta','momentum','size','earning_yield','value','volatility','liquidity','leverage','growth']

obj1 = pick_alpha.pick_alpha(start_date_time,end_date_time,alpha_list,risk_factor_list)
print('done')
# strongest_alpha_number = obj1.strongest_alpha_number
# alpha_sign= obj1.alpha_sign
# #
# # start_date_time='20170106'
# # end_date_time = '20180427'
# obj2 = multifactor.multifactor(start_date_time, end_date_time, risk_factor_list,strongest_alpha_number)

# if config.multifactor_save_residual_list==True:
#     residual_list = []
#     for k in strongest_alpha_number:
#         temp = pd.read_csv(config.multifactor_des_path+'/residual%s_tau%s.csv'%(k,config.predict_period),index_col='Unnamed: 0')
#         residual_list.append(temp)
#     coef_all_df = pd.read_csv(config.multifactor_des_path+'/coef_all_df_tau%s.csv'%config.predict_period)
#     del(coef_all_df['Unnamed: 0'])
#     residual_df = pd.read_csv(config.multifactor_des_path+'/residual_df_tau%s.csv'%config.predict_period,index_col='Unnamed: 0')
# # else:
# #     residual_list = obj2.residual_list
# #     coef_all_df = obj2.coef_all_df
# #     residual_df = obj2.residual_df
# #     alpha_sign = obj1.alpha_sign
# obj3 = newopt.opt('20120720','20180829',risk_factor_list,residual_list,alpha_sign,coef_all_df,residual_df)
# obj3 = opt.opt('20170118','20180425',risk_factor_list,residual_list,alpha_sign,coef_all_df,residual_df)
# obj3 = opt_rewrite.opt('20170118','20180425',risk_factor_list,residual_list,alpha_sign,coef_all_df,residual_df)
