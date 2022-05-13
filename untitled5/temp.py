if only_factor==True:#############getDataReady有更新,only_factor==True时，只get因子，不get Return和weight。默认为False。
    data=pd.concat([other_columns_df, self.other_factors['industry']], axis=1, sort=True)
else:                                      # 默认是close，如果是open，则左移一个单位
    if config.predict_period == 0:         # t+0
        Return = self.other_factors['Return'][timelist_columns]
    elif config.predict_period == 1:       # t+1    # TODO：shift period
        Return  = self.other_factors['Return'].shift(periods=-config.predict_period,axis=1)[timelist_columns]
    else:                                  # t+2
        Return = (self.other_factors['Return'].shift(periods=-2, axis=1)[timelist_columns] + 1) * (self.other_factors['Return'].shift(periods=-1, axis=1)[timelist_columns] + 1) - 1

    if config.opt_returnPrice == "open":   # 判断是open，则左移一个单位
        Return = Return.shift(periods = -1,axis = 1)

    # 合成data变量
    data = pd.concat([Return,other_columns_df, self.other_factors['industry'], self.other_factors['weight'][timelist_columns]], axis=1,sort=True)