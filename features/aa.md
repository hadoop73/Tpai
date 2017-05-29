


- ffm_data.py 用于把数据转换为 ffm 格式

- data_dummy.py 用于 one-hot 数据

- feature_ratio.py 用于产生类别转化率的特征，这里的train_xgbA.csv在xgb中，valid_xgbA.csv
的logloss为0.117

- data_features.py 用于产生特征

- day_features_ratio.py 用于产生每天的转化率，也就是可以统计一天前，两天前，三天前的数据

- user_features.py 用于统计其他类别特征与 appID 组合后的转化率统计

- time_features.py 用于统计1,2,3小时之前各个类别的转化率
