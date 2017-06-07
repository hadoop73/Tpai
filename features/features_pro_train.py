# coding:utf-8


import pandas as pd
import numpy as np
import gc,os
from sklearn.model_selection import train_test_split
from multiprocessing import Pool


if os.path.exists('../data/dup/allAA.csv'):
    d = pd.read_csv('../data/dup/allA.csv')
else:
    train = pd.read_csv('../data/pre/train.csv')
    train.drop(['conversionTime'],axis=1,inplace=True)

    test = pd.read_csv('../data/dup/test.csv')
    test.drop('instanceID',axis=1,inplace=True)

    d = pd.concat([train,test])
    del train,test

    ad = pd.read_csv('../data/dup/ad.csv')
    user = pd.read_csv('../data/dup/user.csv')
    position = pd.read_csv('../data/dup/position.csv')
    d = d.merge(ad,on='creativeID',how='left')
    del ad
    d = d.merge(user,on='userID',how='left')
    del user
    d = d.merge(position,on='positionID',how='left')
    del position

    d.loc[:,'hour'] = d['clickTime'].apply(lambda x:int(x%10000/100))
    # 按天统计
    d.loc[:,'day'] = d['clickTime'].apply(lambda x:int(x/10000))

    d.loc[:,'home'] = d['hometown'].apply(lambda x:int(x/100))
    d.loc[:,'resid'] = d['residence'].apply(lambda x:int(x/100))



    def func(dt):
        dt = dt.reset_index(drop=True)
        n = dt.shape[0]
        for c in ['appID', 'creativeID', 'positionID', 'userID']:
            dt.loc[:, c + 'red'] = 0
            for i in range(1, n):
                dt.loc[i, c + 'red'] = (1 if dt.iloc[i][c] == dt.iloc[i - 1][c] else 0)

        dt.loc[:,'rank'] = range(1,n+1)
        for c in ['appID','creativeID','positionID']:
            ids = pd.unique(dt[c])

            for id in ids:
                n = dt[dt[c]==id].shape[0]
                dt.loc[dt[c]==id,c+'rank'] = range(1,n+1)

        return dt

    days = [d[d['day']==i] for i in range(17,32)]
    pool = Pool(4)
    rst = pool.map(func,days)
    d = pd.concat(rst)
    print d.head()


    d.to_csv('../data/dup/all.csv',index=None)

