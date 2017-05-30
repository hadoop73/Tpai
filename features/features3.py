# coding:utf-8

import pandas as pd
import numpy as np
import gc

train = pd.read_csv('../data/pre/train.csv')
train.drop(['conversionTime'], axis=1, inplace=True)

#print train.head()

train = train[(train['clickTime'] >= 260000)]

user = pd.read_csv('../data/dup/user.csv')
train = train.merge(user, on='userID', how='left')

ad = pd.read_csv('../data/pre/ad.csv')
train = train.merge(ad, on='creativeID', how='left')

train.loc[:, 'clickTime'] = train['clickTime'].apply(lambda x: int(x / 10000))

def fea2():
    adCols = ['adID','creativeID','camgaignID','appID','appPlatform','advertiserID']
    userCols = ['gender','education','marriageStatus','haveBaby']
    for uc in ['creativeID']:
            d = train[userCols+['label','clickTime']]
            d = d.groupby(userCols+['clickTime'],as_index=False)['label'].agg({"{0}_{1}_ratio".format('userCols',uc):np.mean,
                                                                            "{0}_{1}_count".format('userCols', uc):np.sum})
            d.rename(columns={'label':"{0}_{1}_ratio".format('userCols',uc)},inplace=True)
            d.to_csv("../data/dup/{0}_{1}_ratio.csv".format('userCols', uc),index=None)
            print d.head()

fea2()
adCols = ['creativeID', 'camgaignID', 'appID', 'advertiserID']
# adCols = ['adID', 'creativeID', 'camgaignID', 'appID', 'appPlatform', 'advertiserID']

del train


train = pd.read_csv('../data/dup/train_2.csv')
valid = pd.read_csv('../data/dup/valid_2.csv')
test = pd.read_csv('../data/dup/test_2.csv')

userCols = ['gender', 'education', 'marriageStatus', 'haveBaby']

for uc in ['creativeID']:
        d = pd.read_csv("../data/dup/{0}_{1}_ratio.csv".format('userCols', uc))
        d.loc[:, 'clickTime'] = d['clickTime'] + 1
        t = []
        for i in range(3):
            d.loc[:,'clickTime'] = d['clickTime'] + 1
            t.append(d.copy())
        t = pd.concat(t)
        t = t.groupby(userCols+['clickTime'],as_index=False)["{0}_{1}_ratio".format('userCols',uc),"{0}_{1}_count".format('userCols', uc)].mean()
        train = train.merge(t,on=userCols+['clickTime'],how='left')
        train.rename(columns={"{0}_{1}_ratio".format('userCols', uc):"{0}_{1}{2}_ratio".format('userCols', uc,i+1)}, inplace=True)

        valid = valid.merge(t, on=userCols+[ 'clickTime'], how='left')
        valid.rename(columns={"{0}_{1}_ratio".format('userCols', uc): "{0}_{1}{2}_ratio".format('userCols', uc, i + 1)}, inplace=True)

        test = test.merge(t, on=userCols+['clickTime'], how='left')
        test.rename(columns={"{0}_{1}_ratio".format('userCols', uc):"{0}_{1}{2}_ratio".format('userCols', uc,i+1)}, inplace=True)
        del d,t
        gc.collect()

train.fillna(0,inplace=True)
valid.fillna(0,inplace=True)
test.fillna(0,inplace=True)
from train_sample import dataTrainValid

#train,valid = dataTrainValid(train,rate=0.35)

#print train.head()
print test.head()
test.to_csv('../data/dup/test_3.csv',index=None)

train.to_csv('../data/dup/train_3.csv',index=None)
valid.to_csv('../data/dup/valid_3.csv',index=None)

print train.shape,valid.shape,test.shape
