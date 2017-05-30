# coding:utf-8

import pandas as pd
import numpy as np
import gc

#train = pd.read_csv('../data/pre/train.csv')
#train.drop(['conversionTime'], axis=1, inplace=True)

# print train.head()

#train = train[(train['clickTime'] >= 260000)]

test = pd.read_csv('../data/dup/test.csv')
test.drop('instanceID', axis=1, inplace=True)

# d = pd.concat([train,test])
# del train,test

user = pd.read_csv('../data/dup/user.csv')
#train = train.merge(user, on='userID', how='left')
test = test.merge(user, on='userID', how='left')

ad = pd.read_csv('../data/pre/ad.csv')
#train = train.merge(ad, on='creativeID', how='left')
test = test.merge(ad, on='creativeID', how='left')

#train.loc[:, 'clickTime'] = train['clickTime'].apply(lambda x: int(x / 10000))
test.loc[:, 'clickTime'] = test['clickTime'].apply(lambda x: int(x / 10000))

del user,ad

def fea2():
    adCols = ['adID','creativeID','camgaignID','appID','appPlatform','advertiserID']
    userCols = ['gender','education','marriageStatus','haveBaby']
    for ac in adCols:
        for uc in userCols:
            d = train[['label',ac,uc,'clickTime']]
            d = d.groupby([ac,uc,'clickTime'],as_index=False)['label'].agg({"{0}_{1}_ratio".format(ac,uc):np.mean,
                                                                            "{0}_{1}_count".format(ac, uc):np.sum})
            d.rename(columns={'label':"{0}_{1}_ratio".format(ac,uc)},inplace=True)
            d.to_csv("../data/dup/{0}_{1}_ratio.csv".format(ac, uc),index=None)
            print d.head()


adCols = ['creativeID', 'camgaignID', 'appID', 'advertiserID']
# adCols = ['adID', 'creativeID', 'camgaignID', 'appID', 'appPlatform', 'advertiserID']


userCols = ['gender', 'education', 'marriageStatus', 'haveBaby']

for ac in adCols:
    for uc in userCols:
        d = pd.read_csv("../data/dup/{0}_{1}_ratio.csv".format(ac, uc))
        d.loc[:, 'clickTime'] = d['clickTime'] + 1
        t = []
        for i in range(3):
            d.loc[:,'clickTime'] = d['clickTime'] + 1
            t.append(d.copy())
        t = pd.concat(t)
        t = t.groupby([ac,uc,'clickTime'],as_index=False)["{0}_{1}_ratio".format(ac,uc),"{0}_{1}_count".format(ac, uc)].mean()
        #train = train.merge(t,on=[ac,uc,'clickTime'],how='left')
        #train.rename(columns={"{0}_{1}_ratio".format(ac, uc):"{0}_{1}{2}_ratio".format(ac, uc,i+1)}, inplace=True)

        test = test.merge(t, on=[ac,uc,'clickTime'], how='left')
        test.rename(columns={"{0}_{1}_ratio".format(ac, uc):"{0}_{1}{2}_ratio".format(ac, uc,i+1)}, inplace=True)
        del d,t
        gc.collect()

#train.fillna(0,inplace=True)
test.fillna(0,inplace=True)
from train_sample import dataTrainValid

#train,valid = dataTrainValid(train,rate=0.35)

#print train.head()
print test.head()
test.to_csv('../data/dup/test_2.csv',index=None)

#train.to_csv('../data/dup/train_2.csv',index=None)
#valid.to_csv('../data/dup/valid_2.csv',index=None)

#print train.shape,valid.shape,test.shape
