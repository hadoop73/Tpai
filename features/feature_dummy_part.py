# coding:utf-8

import pandas as pd
import numpy as np

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

for f in ['train','valid','test']:

    train = pd.read_csv('../data/dup/{}_r.csv'.format(f))
    #valid = pd.read_csv('../data/dup/valid_r.csv')
    #test = pd.read_csv('../data/dup/test_r.csv')

    #train = pd.concat([train,valid,test])


    d = pd.get_dummies(train['gender'],prefix='gender')
    #train.drop('gender',axis=1,inplace=True)
    train = train.join(d)

    def ageFun(x):
        if x==0: return 0
        if x>10 and x<=20:
            return 1
        elif x <= 30:
            return 2
        elif x==10 or x <=40:
            return 3
        return 4
    train.loc[:,'ageCate'] = train['age'].apply(ageFun)
    d = pd.get_dummies(train['ageCate'],prefix='age')
    train.drop('ageCate',axis=1,inplace=True)
    train = train.join(d)

    def educationFun(x):
        if x>4:
            return 5
        return x
    train.loc[:,'educationCate'] = train['education'].apply(educationFun)
    d = pd.get_dummies(train['educationCate'],prefix='education')
    train.drop('educationCate',axis=1,inplace=True)
    train = train.join(d)

    d = pd.get_dummies(train['marriageStatus'],prefix='marriageStatus')
    train = train.join(d)


    def haveBabyFun(x):
        if x>1:
            return 2
        return x
    train.loc[:,'haveBabyCate'] = train['haveBaby'].apply(haveBabyFun)
    d = pd.get_dummies(train['haveBabyCate'],prefix='haveBaby')
    train.drop('haveBabyCate',axis=1,inplace=True)
    train = train.join(d)

    def creativeIDFun(x):
        if x==4565:
            return 4565
        if x==376:
            return 376
        return 1
    train.loc[:,'creativeIDCate'] = train['creativeID'].apply(creativeIDFun)
    d = pd.get_dummies(train['creativeIDCate'],prefix='creativeID')
    train.drop('creativeIDCate',axis=1,inplace=True)
    train = train.join(d)

    def adIDFun(x):
        cx = [293,3379,3593,3102]
        if x in cx:
            return x
        return 1
    train.loc[:,'adIDCate'] = train['adID'].apply(adIDFun)
    d = pd.get_dummies(train['adIDCate'],prefix='adID')
    train.drop('adIDCate',axis=1,inplace=True)
    train = train.join(d)

    def camgaignIDFun(x):
        cx = [632,649,411,440,201]
        if x in cx:
            return x
        return 1
    train.loc[:,'camgaignIDCate'] = train['camgaignID'].apply(camgaignIDFun)
    d = pd.get_dummies(train['camgaignIDCate'],prefix='camgaignID')
    train.drop('camgaignIDCate',axis=1,inplace=True)
    train = train.join(d)

    def advertiserIDFun(x):
        cx = [3,84,81,15,89]
        if x in cx:
            return x
        return 1
    train.loc[:,'advertiserIDCate'] = train['advertiserID'].apply(advertiserIDFun)
    d = pd.get_dummies(train['advertiserIDCate'],prefix='advertiserID')
    train.drop('advertiserIDCate',axis=1,inplace=True)
    train = train.join(d)


    d = pd.get_dummies(train['appPlatform'],prefix='appPlatform')
    train = train.join(d)


    def appIDFun(x):
        cx = [465,360,421,109,383]
        if x in cx:
            return x
        return 1
    train.loc[:,'appIDCate'] = train['appID'].apply(appIDFun)
    d = pd.get_dummies(train['appIDCate'],prefix='appID')
    train.drop('appIDCate',axis=1,inplace=True)
    train = train.join(d)


    d = pd.get_dummies(train['telecomsOperator'],prefix='telecomsOperator')
    train = train.join(d)


    def connectionTypeFun(x):
        cx = [465,360,421,109,383]
        if x in cx:
            return x
        return 1
    train.loc[:,'connectionTypeCate'] = train['connectionType'].apply(connectionTypeFun)
    d = pd.get_dummies(train['connectionTypeCate'],prefix='connectionType')
    train.drop('connectionTypeCate',axis=1,inplace=True)
    train = train.join(d)


    train.loc[:,'resid'] = train['residence'].apply(lambda x:int(x/100))
    train.loc[:,'home'] = train['hometown'].apply(lambda x:int(x/100))


    def homeFun(x):
        if x==0: return 0
        cx = [1,2,3,4,6]
        if x in cx:
            return 2
        return 1
    train.loc[:,'homeCate'] = train['home'].apply(homeFun)
    d = pd.get_dummies(train['homeCate'],prefix='home')
    train.drop('homeCate',axis=1,inplace=True)
    train = train.join(d)

    def residFun(x):
        if x < 4:
            return x
        return 1
    train.loc[:,'residCate'] = train['resid'].apply(residFun)
    d = pd.get_dummies(train['residCate'],prefix='resid')
    train.drop('residCate',axis=1,inplace=True)
    train = train.join(d)
    print train.shape
    train.to_csv('../data/dup/{}_d.csv'.format(f),index=None)
    del train

"""
valid = train[(train['clickTime']>=290000)&(train['clickTime']<300000)]
test = train[(train['clickTime']>=310000)]

train = train[(train['clickTime']<290000)]

print train.head()
print train.shape,valid.shape,test.shape

train.to_csv('../data/dup/train_dm.csv',index=None)
valid.to_csv('../data/dup/valid_dm.csv',index=None)
test.to_csv('../data/dup/test_dm.csv',index=None)
"""
