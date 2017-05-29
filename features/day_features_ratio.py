# coding:utf-8


import pandas as pd
import numpy as np


train = pd.read_csv('../data/dup/train.csv')
train.drop('conversionTime',axis=1,inplace=True)

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

# 按天统计
d.loc[:,'clickTime'] = d['clickTime'].apply(lambda x:int(x/10000))

d.loc[:,'hometown'] = d['hometown'].apply(lambda x:int(x/100))
d.loc[:,'residence'] = d['residence'].apply(lambda x:int(x/100))

d = d[d['clickTime']>=21]

cols = ['gender','education','marriageStatus','haveBaby','hometown','residence','sitesetID',
        'adID','camgaignID','appID','appPlatform','creativeID','advertiserID','positionID','positionType',
        'connectionType','telecomsOperator']
import gc

for col in cols:
    t = d[['label','clickTime',col]]
    t = t.groupby([col,'clickTime'],as_index=False)['label'].mean()
    t.rename(columns={'label':col+"day_ratio"},inplace=True)
    for i in range(3):
        t.loc[:,'clickTime'] = t['clickTime'] + 1
        d = d.merge(t,on=[col,'clickTime'],how='left')
        d.rename(columns={col+"day_ratio":col+str(i+1)+"day_ratio"},inplace=True)
        gc.collect()


from train_sample import dataSampleDay
train,valid,test = dataSampleDay(d,rate=0.2)

print train.shape,valid.shape,test.shape

train.to_csv('../data/dup/train_xgbD133.csv',index=None)
valid.to_csv('../data/dup/valid_xgbD133.csv',index=None)
test.to_csv('../data/dup/test_xgbD133.csv',index=None)


