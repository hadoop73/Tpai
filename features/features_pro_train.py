# coding:utf-8


import pandas as pd
import numpy as np
import gc,os
from sklearn.model_selection import train_test_split
from multiprocessing import Pool


if os.path.exists('../data/dup/allA.csv'):
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


    for day in range(17,32):
        dt = d[(d['day'] >= day) & (d['day'] < day+1)]
        n = dt.shape[0]
        d.loc[(d['day']>=day)&(d['day']<day+1),'rank'] = range(1,n+1)
        creativeIDs = pd.unique(dt['creativeID'])

        for id in creativeIDs:

            n = dt[dt['creativeID']==id].shape[0]
            d.loc[(d['day']>=day)&(d['day']<day+1)&(d['creativeID']==id),'creativeRank'] = range(1,n+1)

    print d.head()


    d.to_csv('../data/dup/allA.csv',index=None)

d.loc[:,'cID'] = 0
d.loc[(d['creativeID']==4565),'cID'] = 1
d.loc[(d['creativeID']==376),'cID'] = 2

d.loc[:,'pID'] = 0
d.loc[(d['positionID']==2579),'pID'] = 1
d.loc[(d['positionID']==3322),'pID'] = 2


print d.head()
print d.shape
d.to_csv('../data/dup/all.csv',index=None)
