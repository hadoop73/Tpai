# coding:utf-8

import pandas as pd
import numpy as np


train = pd.read_csv('../data/dup/train_count.csv')
train.drop(['conversionTime','recode_str'],axis=1,inplace=True)

#print train.head()

train = train[(train['clickTime']>=240000)&(train['clickTime']<290000)]

#test = pd.read_csv('../data/dup/test.csv')
#test.drop('instanceID',axis=1,inplace=True)

#d = pd.concat([train,test])
#del train,test

user = pd.read_csv('../data/dup/user.csv')
train = train.merge(user,on='userID',how='left')

adCols = ['adID','creativeID','camgaignID','appID','appPlatform','advertiserID']


userCols = ['gender','education','marriagedStatus','have']

for ac in adCols:
    for uc in userCols:
        d = train[['label',ac,uc]]
        d = d.groupby([ac,uc],as_index=False)['label'].mean()
        d.rename('label',ac+uc+'ratio',inplace=True)
        train = train.merge(d,on=[ac,uc],how='left')


train.to_csv('../data/dup/train2.csv',index=None)
print train.head()

