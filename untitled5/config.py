
import pandas as pd
import os


# ------------------------------ document path ------------------------------ #
# 获取当前程序的地址
current_file = __file__

# 程序根目录地址，os.pardir：父目录
root_path = os.path.abspath(os.path.join(current_file, os.pardir, os.pardir))

# 输入数据根目录地址
input_data_path = os.path.abspath(os.path.join(root_path, 'file', 'input'))
file_path = os.path.abspath(os.path.join(root_path, 'file'))
# 输出数据根目录地址
output_data_path = os.path.abspath(os.path.join(root_path, 'file', 'output'))

dir_path = os.path.abspath(os.path.join(root_path, 'file', 'fun_pq_factors'))
risk_path = os.path.abspath(os.path.join(root_path, 'file', 'risk_factors'))
other_factor_path = os.path.abspath(os.path.join(root_path, 'file', 'other_factors'))
alpha_path = os.path.abspath(os.path.join(root_path, 'file', 'alpha'))
# dir_path = os.path.dirname(os.path.realpath(__file__))
W_BM = pd.read_csv('/Users/YoYo/Desktop/无隅资管/SVN/SVN/risk_new/W_bm_zz_500.csv',index_col=0)*100 #benchmark
# risk_path = '/Users/YoYo/Desktop/无隅资管/SVN/SVN/risk_new/'
# other_factor_path = risk_path
# alpha_path = '/Users/YoYo/Desktop/无隅资管/SVN/SVN/alpha/'
# #output_path = dir_path + '/output/'
# output_path = '/Users/YoYo/Desktop/无隅资管/SVN/SVN/output/'
# summary_path = output_path
# draw_picture_path = output_path
# multifactor_des_path= output_path #save residual list path
# opt_des_path= output_path  ###save统计指标的excel


# ------------------------------ 单因子有效性检验参数 ------------------------------ #

is_test_predict_period = True
predict_period = 2
annual_return_threshold = 0.04
IR_threshold = 1.5
t_mean_abs_threshold = 2.0
t_ratio_threshold = 0.5
alpha_correlation_threshold = 0.8

# ------------------------------ 多因子回归参数 ------------------------------ #

multifactor_window = 250
multifactor_save_residual_list=True
iter_period = 500
industry_name_list = ['交通运输','休闲服务','传媒','公用事业','农林牧渔','化工','医药生物','商业贸易',
                      '国防军工','家用电器','建筑材料','建筑装饰','房地产开发Ⅱ','园区开发Ⅱ',
                      '有色金属','机械设备','汽车', '电子','电气设备','纺织服装','综合','计算机',
                      '轻工制造','通信','采掘','钢铁','银行','保险Ⅱ','证券Ⅱ','多元金融Ⅱ',
                      '食品加工','饮料制造']

# ------------------------------ 权重优化参数 ------------------------------ #

opt_ls_strategy = 'ls'   # 'lbm', 'sbm', 'ls'
opt_window = 250
opt_te=0.12
opt_obj_tc = 0.00
opt_tc=0.00
# opt_exposure = 1  # =1是放开风险敞口，=0是不放开
#beta,momentum,size,earning_yield,value,volatility,liquidity,leverage,growth
opt_risk_upperbound=[0.01, 100, 100, 100,  0.01,-0.2, 0.01, 0.01, 0.01]
opt_risk_lowerbound=[-0.01, 0.3, -0.3, 0.5, -0.01, -100, -0.01, -0.01, -0.01]
opt_risk_upperbound=[100, 100, 100, 100,  100, 100, 100, 100, 100]
opt_risk_lowerbound=[-100, -100, -0.3, -100, -100, -100, -100, -100, -100]
opt_risk_upperbound=[0.1, 0.1, 100, 100,  100, 0.1, 100, 0.1, 100]
opt_risk_lowerbound=[-0.1, -0.1, -0.1, -100, -100, -0.1, -100, -0.1, -100] # 对风险因子的风险敞口控制
opt_risk_upperbound=[0.01, 100, 100, 0.01,  -0.3, 0.01, 0.01, 0.01]
opt_risk_lowerbound=[-0.01, 0.3, 0.5, -0.01, -100, -0.01, -0.01, -0.01]
# opt_risk_upperbound=[0.01, 0.1, 0.1, 0.1,  0.01, 0.1, 0.01, 0.01, 0.01]
# opt_risk_lowerbound=[-0.01, -0.1, -0.1, -0.1, -0.01, -0.1, -0.01, -0.01, -0.01]
opt_industry_upperbound=0.02
opt_industry_lowerbound=-0.02
opt_stock_upperbound = 0.03
opt_stock_lowerbound = 0.003 #只适用于lbm 和 sbm两种情况，不适应于ls
opt_exp_method = 250 #可选：'mv'/60、250等正整数/固定值：0(0.0001)
opt_pkg = 'cpy' #可用 cpy（指cvxpy）, mnl(指manual), #不可用： msk(指mosek), cpt(指cvxopt)
opt_method = 'abs'   #可选：'abs'/ 4/3 /2
opt_object_function = 0 # =1是目标函数里包含风险，=0是不包含风险。目前cvxpy可用。
opt_object_lambda = 1.5 #目标函数风险前的系数, 风险厌恶系数
opt_expectreturn_lambda = 0.0 #fector expect return
opt_constraint = 1 # =1是限制条件包含风险，=0是限制条件不包含风险。目前cvxpy和cvxopt可用。
opt_cvxpy_solver = 'SCS' # ECOS...
opt_exaggerate_times=10000
opt_factor_weight_method = 'opw' # opw: optimized weight /eqw: equal weight for alpha+style factors in obj fun
opt_factor_weight = 49  # total weight of alpha+style in obj fun
opt_factor_include = 'alpha' # 可选：'alpha' or 'both'
opt_eps = 1e-3
opt_interval = 2
opt_days = 1
opt_returnPrice = 'open' # 'open','close'
weight_upperbound=1.05
weight_lowerbound=0.95

# ------------------------------ calculate return 参数 ------------------------------ #

opt_returnPrice = 'open'




