#!/home/hadoop/env2.7/bin/python
# coding:utf-8


import argparse
import pandas as pd
import numpy as np

"""
    特征产生思路：1) 先用 train21all.csv 产生所有数据的特征
                2) 再用 train21.csv 的数据与 train21all.csv 的特征进行 join 合并
                3) 对 valid test 同样可以通过步骤 2) 获得特征
"""


"""
1 获得 appID 的激活情况
"""

#train = pd.read_csv('../data/dup/train21all.csv')

def user_live_state():
    user = pd.read_csv('../data/dup/user.csv')
    user.loc[(0 == user['age']), 'age'] = -10

    user['zeroSum'] = (user==0).sum(axis=1)
    # hometown 和 residence 是否相同
    user.loc[(user['hometown']==user['residence']),'liveState'] = 1
    user.loc[(user['hometown']!=user['residence']),'liveState'] = 0
    user.loc[(0 == user['residence']), 'liveState'] = -10 # 在 residence 缺失，livestate
    user['residenceliveState'] = user['residence'].apply(lambda x: x==0 and 0 or 1)

    user['residence'] = user['residence'].apply(lambda x: int(x / 100))
    user['hometown'] = user['hometown'].apply(lambda x: int(x / 100))

    print user.head()
    user.to_csv('../data/dup/user_state.csv',index=None)

rand = 11
from train_sample import trainDay
def xx():
    d = pd.read_csv('../data/dup/user_state.csv')
    d = d[['userID','zeroSum','liveState']]

    train = pd.read_csv('../data/dup/train_xgbM{}.csv'.format(rand))
    train = train.merge(d,on='userID',how='left')

    valid = pd.read_csv('../data/dup/valid_xgbM{}.csv'.format(rand))
    valid = valid.merge(d,on='userID',how='left')

    test = pd.read_csv('../data/dup/test_xgbM{}.csv'.format(rand))
    test = test.merge(d, on='userID', how='left')

    trainDf = pd.read_csv('../data/dup/train.csv')
    df = trainDf[['label', 'userID','clickTime']]
    df['clickTimeDay'] = df['clickTime'].apply(lambda x: int(x / 10000))

    df = df.merge(d, on='userID', how='left')
    del trainDf

    for col in ['liveState','zeroSum']:
        d = df[['label',col,'clickTimeDay']]

        ts = []
        for i in range(1, 6):
            td = d.copy()
            td.loc[:, 'clickTimeDay'] = td['clickTimeDay'].apply(lambda x: x + i)
            ts.append(td)
            ds = pd.concat(ts)
            # print ds.head()
            ds = ds.groupby([col, 'clickTimeDay'], as_index=False)['label'].agg({col + "_ratio_" + str(i): np.mean,
                                                                                 col + "_size_" + str(i): np.size,
                                                                                 col + "_sum_" + str(i): np.sum})
            ds.loc[:, col + "_size_" + str(i)] = ds[col + "_size_" + str(i)].apply(lambda x: x / i)
            ds.loc[:, col + "_sum_" + str(i)] = ds[col + "_sum_" + str(i)].apply(lambda x: x / i)

            # train29 = train29.merge(d, on=[col,'clickTimeDay'], how='left')
            train = train.merge(ds, on=[col, 'clickTimeDay'], how='left')
            valid = valid.merge(ds, on=[col, 'clickTimeDay'], how='left')
            test = test.merge(ds, on=[col, 'clickTimeDay'], how='left')

    train.to_csv('../data/dup/train_xgb{}U.csv'.format(rand), index=None)
    valid.to_csv('../data/dup/valid_xgb{}U.csv'.format(rand), index=None)
    test.to_csv('../data/dup/test_xgb{}U.csv'.format(rand), index=None)
    print train.shape, valid.shape, test.shape


"""
    1 统计 user_features 特征
"""
def mergeUserDatas():
    train = pd.read_csv('../data/dup/train_xgb{}U.csv'.format(rand))
    valid = pd.read_csv('../data/dup/valid_xgb{}U.csv'.format(rand))
    test = pd.read_csv('../data/dup/test_xgb{}U.csv'.format(rand))
    #train = pd.read_csv('../data/dup/train21all_feas.csv')
    # 减少数据量，筛选 train，valid，test
    #train = train[((train['clickTime']>=240000)&(train['clickTime']<280000))|((train['clickTime'] >= 290000) & (train['clickTime'] < 300000))|(train['clickTime']>=310000)]

    print train['clickTime'].min(), train['clickTime'].max()
    #return
    cols = ['gender','education','marriageStatus','haveBaby','hometown',
                'residence','zeroSum','liveState','connectionType',
                'telecomsOperator','clickTime']

    train.loc[:,'clickTimecp'] = train['clickTime']
    train.loc[:,'clickTime'] = train['clickTime'].apply(lambda x:int(x%10000/100))

    valid.loc[:, 'clickTimecp'] = valid['clickTime']
    valid.loc[:, 'clickTime'] = valid['clickTime'].apply(lambda x: int(x % 10000 / 100))

    test.loc[:, 'clickTimecp'] = test['clickTime']
    test.loc[:, 'clickTime'] = test['clickTime'].apply(lambda x: int(x % 10000 / 100))

    for col in cols:
        d = pd.read_csv('../data/user/appID_{}.csv'.format(col))
        train = train.merge(d,on=['appID',col],how='left')
        valid = valid.merge(d, on=['appID', col],how='left')
        test = test.merge(d, on=['appID', col],how='left')

    train.loc[:, 'clickTime'] = train['clickTimecp']
    valid.loc[:, 'clickTime'] = valid['clickTimecp']
    test.loc[:, 'clickTime'] = test['clickTimecp']

    train.drop('clickTimecp',axis=1,inplace=True)
    valid.drop('clickTimecp',axis=1,inplace=True)
    test.drop('clickTimecp', axis=1, inplace=True)

    train.to_csv('../data/dup/train_data_user{}.csv'.format(rand),index=None)
    valid.to_csv('../data/dup/valid_data_user{}.csv'.format(rand), index=None)
    test.to_csv('../data/dup/test_data_user{}.csv'.format(rand), index=None)

    print train.shape,valid.shape,test.shape


def mergeDataHour():
    #train = pd.read_csv('../data/dup/data_user.csv')
    train = pd.read_csv('../data/dup/train_xgb{}U.csv'.format(rand))
    valid = pd.read_csv('../data/dup/valid_xgb{}U.csv'.format(rand))
    test = pd.read_csv('../data/dup/test_xgb{}U.csv'.format(rand))

    train.loc[:, 'clickTimecp'] = train['clickTime']
    train.loc[:,'clickTime'] = train['clickTimecp'].apply(lambda x:int(24*(x/10000-17))+int(x%10000/100))

    valid.loc[:, 'clickTimecp'] = valid['clickTime']
    valid.loc[:,'clickTime'] = valid['clickTimecp'].apply(lambda x:int(24*(x/10000-17))+int(x%10000/100))

    test.loc[:, 'clickTimecp'] = test['clickTime']
    test.loc[:, 'clickTime'] = test['clickTimecp'].apply(lambda x: int(24 * (x / 10000 - 17)) + int(x % 10000 / 100))
    #train.loc[:,'clickTime'] = train['clickTimecp'].apply(lambda x:int(24*(x/10000-17))+int(x%10000/100))
    print train.head()
    # 合并几个小时之前的统计数据量

    import gc

    cols = ['gender', 'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence',
            'adID', 'camgaignID', 'appID', 'appPlatform', 'creativeID', 'advertiserID',
            'connectionType', 'telecomsOperator', 'liveState']

    for x in range(5):
        train.loc[:, 'clickTime'] = train['clickTime'] - 1
        valid.loc[:, 'clickTime'] = valid['clickTime'] - 1
        test.loc[:, 'clickTime'] = test['clickTime'] - 1
        for col in cols:
            d = pd.read_csv('../data/time/onehour_{}.csv'.format(col))
            train = train.merge(d,on=[col,'clickTime'],how='left')
            valid = valid.merge(d, on=[col, 'clickTime'], how='left')
            test = test.merge(d, on=[col, 'clickTime'], how='left')

            gc.collect()
            train.rename(columns={col+'_size_hour':col+'_size_hour'+str(x+1)},inplace=True)
            valid.rename(columns={col+'_size_hour':col+'_size_hour'+str(x+1)}, inplace=True)
            test.rename(columns={col+'_size_hour':col+'_size_hour'+str(x+1)}, inplace=True)

    train.loc[:, 'clickTime'] = train['clickTimecp']
    valid.loc[:, 'clickTime'] = valid['clickTimecp']
    test.loc[:, 'clickTime'] = test['clickTimecp']

    train.drop('clickTimecp', axis=1, inplace=True)
    valid.drop('clickTimecp', axis=1, inplace=True)
    test.drop('clickTimecp', axis=1, inplace=True)

    train.to_csv('../data/dup/train_user_hour{}.csv'.format(rand),index=None)
    valid.to_csv('../data/dup/valid_user_hour{}.csv'.format(rand),index=None)
    test.to_csv('../data/dup/test_user_hour{}.csv'.format(rand), index=None)
    print train.shape,valid.shape,test.shape



def mergeDataDay():
    train = pd.read_csv('../data/dup/train_user_hour{}.csv'.format(rand))
    valid = pd.read_csv('../data/dup/valid_user_hour{}.csv'.format(rand))
    test = pd.read_csv('../data/dup/test_user_hour{}.csv'.format(rand))

    train.loc[:, 'clickTimecp'] = train.loc[:, 'clickTime']
    train.loc[:, 'clickTime'] = train['clickTimecp'].apply(lambda x:int(x/100))

    valid.loc[:, 'clickTimecp'] = valid.loc[:, 'clickTime']
    valid.loc[:, 'clickTime'] = valid['clickTimecp'].apply(lambda x: int(x / 100))

    test.loc[:, 'clickTimecp'] = test.loc[:, 'clickTime']
    test.loc[:, 'clickTime'] = test['clickTimecp'].apply(lambda x: int(x / 100))

    import gc
    cols = ['gender', 'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence',
            'adID', 'camgaignID', 'appID', 'appPlatform', 'creativeID', 'advertiserID',
            'connectionType', 'telecomsOperator', 'liveState']
    for x in range(3):
        train.loc[:, 'clickTime'] = train['clickTime'] - 100
        valid.loc[:, 'clickTime'] = valid['clickTime'] - 100
        test.loc[:, 'clickTime'] = test['clickTime'] - 100

        for col in cols:
            d = pd.read_csv('../data/time/days_{}.csv'.format(col))
            train = train.merge(d, on=[col, 'clickTime'], how='left')
            valid = valid.merge(d, on=[col, 'clickTime'], how='left')
            test = test.merge(d, on=[col, 'clickTime'], how='left')

            gc.collect()
            train.rename(columns={col + '_size_day': col + '_size_day' + str(x + 1)}, inplace=True)
            train.rename(columns={col + '_ratio_day': col + '_ratio_day' + str(x + 1)}, inplace=True)
            train.rename(columns={col + '_sum_day': col + '_sum_day' + str(x + 1)}, inplace=True)

            valid.rename(columns={col + '_size_day': col + '_size_day' + str(x + 1)}, inplace=True)
            valid.rename(columns={col + '_ratio_day': col + '_ratio_day' + str(x + 1)}, inplace=True)
            valid.rename(columns={col + '_sum_day': col + '_sum_day' + str(x + 1)}, inplace=True)

            test.rename(columns={col + '_size_day': col + '_size_day' + str(x + 1)}, inplace=True)
            test.rename(columns={col + '_ratio_day': col + '_ratio_day' + str(x + 1)}, inplace=True)
            test.rename(columns={col + '_sum_day': col + '_sum_day' + str(x + 1)}, inplace=True)

            #valid.rename(columns={col + '_ratio_day': col + '_ratio_day' + str(x + 1)}, inplace=True)
            #test.rename(columns={col + '_sum_day': col + '_sum_day' + str(x + 1)}, inplace=True)


    train.loc[:, 'clickTime'] = train['clickTimecp']
    valid.loc[:, 'clickTime'] = valid['clickTimecp']
    test.loc[:, 'clickTime'] = test['clickTimecp']

    train.drop('clickTimecp', axis=1, inplace=True)
    valid.drop('clickTimecp', axis=1, inplace=True)
    test.drop('clickTimecp', axis=1, inplace=True)

    train.to_csv('../data/dup/train_user_hd{}.csv'.format(rand), index=None)
    valid.to_csv('../data/dup/valid_user_hd{}.csv'.format(rand), index=None)
    test.to_csv('../data/dup/test_user_hd{}.csv'.format(rand),index=None)
    print train.shape, valid.shape, test.shape

xx()
#mergeUserDatas()
#mergeDataHour()
#mergeDataDay()
#mergeDataDayActive()



