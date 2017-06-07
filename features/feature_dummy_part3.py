# coding:utf-8

import pickle
import pandas as pd
import numpy as np



cols = ['age','home','hometown','residence','resid','positionID',
        'creativeID','adID','camgaignID','advertiserID']


train = pd.read_csv('../data/dup/train.csv')
train = train[(train['clickTime']>=210000)&((train['clickTime']<300000))]

user = pd.read_csv('../data/dup/user.csv')
train = train.merge(user,on='userID',how='left')
ad = pd.read_csv('../data/dup/ad.csv')
train = train.merge(ad,on='creativeID',how='left')
position = pd.read_csv('../data/dup/position.csv')
train = train.merge(position,on='positionID',how='left')
del user,ad,position

train.loc[:,'resid'] = train['residence'].apply(lambda x:int(x/100))
train.loc[:,'home'] = train['hometown'].apply(lambda x:int(x/100))
print train.head()

cateDatas = {}

for c in cols:
    d = train[[c, 'label']]
    dt = pd.pivot_table(index=c, columns='label', data=d, aggfunc=np.size).reset_index()
    dt.sort_values(by=[1], ascending=False, inplace=True)
    cateDatas[c] = dt[c].values[:50]
    print c,cateDatas[c]

f = open('../data/dup/cateDatas.dict','wb')
pickle.dump(cateDatas,f)
