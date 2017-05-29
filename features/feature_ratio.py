# coding:utf-8

import pandas as pd
import numpy as np
from train_sample import trainSample
import gc
# 统计从 22-30 这些天之前 1,2,3,5 天的转化率

def colDayRatio(rand=133):
    train29 = pd.read_csv('../data/dup/train.csv')
    train29.drop('conversionTime', axis=1, inplace=True)
    #train29 = train29[train29['clickTime']<310000]
    train,valid,test = trainSample(rand=rand)
    print train.shape, valid.shape, test.shape

    ad = pd.read_csv('../data/dup/ad.csv')
    train29 = train29.merge(ad, on='creativeID', how='left')
    train = train.merge(ad,on='creativeID',how='left')
    valid = valid.merge(ad, on='creativeID', how='left')
    test = test.merge(ad, on='creativeID', how='left')

    user = pd.read_csv('../data/dup/user.csv')
    train29 = train29.merge(user, on='userID', how='left')
    train = train.merge(user,on='userID',how='left')
    valid = valid.merge(user,on='userID',how='left')
    test = test.merge(user, on='userID', how='left')


    train29.loc[:, 'clickTimeDay'] = train29['clickTime'].apply(lambda x: int(x / 10000 ))
    train.loc[:, 'clickTimeDay'] = train['clickTime'].apply(lambda x: int(x / 10000))
    valid.loc[:, 'clickTimeDay'] = valid['clickTime'].apply(lambda x: int(x / 10000))
    test.loc[:, 'clickTimeDay'] = test['clickTime'].apply(lambda x: int(x / 10000))

    train29.loc[:, 'hometown'] = train29['hometown'].apply(lambda x: int(x / 100))
    train29.loc[:, 'residence'] = train29['residence'].apply(lambda x: int(x / 100))

    train.loc[:,'hometown'] = train['hometown'].apply(lambda x:int(x/100))
    train.loc[:,'residence'] = train['residence'].apply(lambda x:int(x/100))

    valid.loc[:, 'hometown'] = valid['hometown'].apply(lambda x: int(x / 100))
    valid.loc[:, 'residence'] = valid['residence'].apply(lambda x: int(x / 100))

    test.loc[:, 'hometown'] = test['hometown'].apply(lambda x: int(x / 100))
    test.loc[:, 'residence'] = test['residence'].apply(lambda x: int(x / 100))

    cols = ['creativeID','userID','adID','camgaignID','appID','appPlatform','advertiserID',
            'connectionType','telecomsOperator','gender','education','positionID',
            'marriageStatus','haveBaby','hometown','residence']

    """# train_xgbA.csv 也就是 0.117 的数据，主要时统计了平均转化率
    for col in cols:
        d = train[[col,'label']]
        d = d.groupby(col,as_index=False)['label'].mean()
        d.rename(columns={'label':col+'_mean_ratio'},inplace=True)
        train = train.merge(d,on=col,how='left')
        valid = valid.merge(d, on=col, how='left')
        test = test.merge(d, on=col, how='left')
    """
    # 统计平均准化率，以及点击次数，转化次数
    for col in cols:
        d = train29[[col,'clickTimeDay', 'label']]
        ts = []
        for i in range(1,6):
            td = d.copy()
            td.loc[:,'clickTimeDay'] = td['clickTimeDay'].apply(lambda x:x+i)
            ts.append(td)
            ds = pd.concat(ts)
            #print ds.head()
            ds = ds.groupby([col,'clickTimeDay'],as_index=False)['label'].agg({col+"_ratio_"+str(i):np.mean,
                                                                               col + "_size_" + str(i):np.size,
                                                                               col + "_sum_" + str(i):np.sum})
            ds.loc[:,col + "_size_" + str(i)] = ds[col + "_size_" + str(i)].apply(lambda x:x/i)
            ds.loc[:,col + "_sum_" + str(i)] = ds[col + "_sum_" + str(i)].apply(lambda x:x/i)

            ds.rename(columns={'label':col+"_ratio_"+str(i)},inplace=True)
            #train29 = train29.merge(d, on=[col,'clickTimeDay'], how='left')
            train = train.merge(ds, on=[col,'clickTimeDay'], how='left')
            valid = valid.merge(ds, on=[col,'clickTimeDay'], how='left')
            test = test.merge(ds, on=[col,'clickTimeDay'], how='left')
            gc.collect()
    #print train.shape
    print train.shape,valid.shape,test.shape

    #train29.to_csv('../data/dup/train_xgball{}.csv'.format(rand), index=None)
    train.to_csv('../data/dup/train_xgbM{}.csv'.format(rand), index=None)
    #train.to_csv('../data/dup/train_xgbASample.csv', index=None)
    valid.to_csv('../data/dup/valid_xgbM{}.csv'.format(rand), index=None)
    test.to_csv('../data/dup/test_xgbM{}.csv'.format(rand), index=None)


def colRatio(rand=133):
    train29 = pd.read_csv('../data/dup/train.csv')
    train29.drop('conversionTime', axis=1, inplace=True)
    #train29 = train29[train29['clickTime']<310000]
    train,valid,test = trainSample(rand=rand)
    print train.shape, valid.shape, test.shape

    ad = pd.read_csv('../data/dup/ad.csv')
    train29 = train29.merge(ad, on='creativeID', how='left')
    train = train.merge(ad,on='creativeID',how='left')
    valid = valid.merge(ad, on='creativeID', how='left')
    test = test.merge(ad, on='creativeID', how='left')

    user = pd.read_csv('../data/dup/user.csv')
    train29 = train29.merge(user, on='userID', how='left')
    train = train.merge(user,on='userID',how='left')
    valid = valid.merge(user,on='userID',how='left')
    test = test.merge(user, on='userID', how='left')

    train29.loc[:, 'clickTimeCateHour'] = train29['clickTime'].apply(lambda x: int(x % 10000 / 100))
    train.loc[:, 'clickTimeCateHour'] = train['clickTime'].apply(lambda x: int(x % 10000 / 100))
    valid.loc[:, 'clickTimeCateHour'] = valid['clickTime'].apply(lambda x: int(x % 10000 / 100))
    test.loc[:, 'clickTimeCateHour'] = test['clickTime'].apply(lambda x: int(x % 10000 / 100))

    train29.loc[:, 'hometown'] = train29['hometown'].apply(lambda x: int(x / 100))
    train29.loc[:, 'residence'] = train29['residence'].apply(lambda x: int(x / 100))

    train.loc[:,'hometown'] = train['hometown'].apply(lambda x:int(x/100))
    train.loc[:,'residence'] = train['residence'].apply(lambda x:int(x/100))

    valid.loc[:, 'hometown'] = valid['hometown'].apply(lambda x: int(x / 100))
    valid.loc[:, 'residence'] = valid['residence'].apply(lambda x: int(x / 100))

    test.loc[:, 'hometown'] = test['hometown'].apply(lambda x: int(x / 100))
    test.loc[:, 'residence'] = test['residence'].apply(lambda x: int(x / 100))

    cols = ['creativeID','clickTimeCateHour','userID','adID','camgaignID','appID','appPlatform','advertiserID',
            'connectionType','telecomsOperator','gender','education','positionID',
            'marriageStatus','haveBaby','hometown','residence']

    """# train_xgbA.csv 也就是 0.117 的数据，主要时统计了平均转化率
    for col in cols:
        d = train[[col,'label']]
        d = d.groupby(col,as_index=False)['label'].mean()
        d.rename(columns={'label':col+'_mean_ratio'},inplace=True)
        train = train.merge(d,on=col,how='left')
        valid = valid.merge(d, on=col, how='left')
        test = test.merge(d, on=col, how='left')
    """
    # 统计平均准化率，以及点击次数，转化次数
    for col in cols:
        d = train29[[col, 'label']]
        d = d.groupby(col, as_index=False)['label'].agg({col+'_mean_ratio':np.mean,
                                                         col+'_counts':np.size,
                                                         col+'_positivecounts':np.sum})
        train = train.merge(d, on=col, how='left')
        valid = valid.merge(d, on=col, how='left')
        test = test.merge(d, on=col, how='left')
    #train.drop(cols,axis=1,inplace=True)
    #valid.drop(cols, axis=1, inplace=True)

    #train.to_csv('../data/dup/train_all_xgb.csv',index=None)
    #train = train[(train['clickTime'] >= 260000) & (train['clickTime'] < 280000)]
    #train.drop('clickTime', axis=1, inplace=True)
    #valid.drop('clickTime', axis=1, inplace=True)

    print train.shape,valid.shape,test.shape
    train.to_csv('../data/dup/train_xgb{}.csv'.format(rand), index=None)
    #train.to_csv('../data/dup/train_xgbASample.csv', index=None)
    valid.to_csv('../data/dup/valid_xgb{}.csv'.format(rand), index=None)
    test.to_csv('../data/dup/test_xgb{}.csv'.format(rand), index=None)

import time
start = time.time()
colDayRatio(rand=11)


print 'time:',time.time()-start
