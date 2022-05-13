#import multifactor
import pick_alpha
#import newopt


start_date_time = '20090301'
end_date_time = '20180831'
alpha_list = [435]


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