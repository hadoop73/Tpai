# coding:utf-8

import pandas as pd
import numpy as np
import pickle

"""
train = pd.read_csv('../data/pre/train.csv')
ad = pd.read_csv('../data/dup/ad.csv')
position = pd.read_csv('../data/dup/position.csv')
user = pd.read_csv('../data/dup/user.csv')

train = train[train['clickTime']>=220000]
train.drop('conversionTime',axis=1,inplace=True)

test = pd.read_csv('../data/dup/test.csv')
test.drop('instanceID',axis=1,inplace=True)

train = pd.concat([train,test])

train = train.merge(ad,on='creativeID',how='left')
del ad
train = train.merge(position,on='positionID',how='left')
del position
train = train.merge(user,on='userID',how='left')
del user
"""
f = open('../data/dup/cateDatas.dict','rb')
cateDatas = pickle.load(f)

for f in ['train','valid','test']:

    train = pd.read_csv('../data/dup/{}_r2.csv'.format(f))

    cols = ['age', 'home', 'hometown', 'residence', 'resid', 'positionID',
            'creativeID', 'adID', 'camgaignID', 'advertiserID']

    for c in cols:
        if c in train.columns:
            train.loc[:,c] = train[c].apply(lambda x:x if x in cateDatas[c] else 999999)

    train.loc[:,'999999'] = (train==999999).sum(axis=1)
    print train.head()
    print train.shape
    train.to_csv('../data/dup/{}_d.csv'.format(f),index=None)
    del train


