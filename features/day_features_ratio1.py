# coding:utf-8


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import os

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



cols1 = ['home', 'resid', 'positionID','hour','age',
        'creativeID', 'adID', 'camgaignID','advertiserID','appID']

cols = []

n2 = len(cols1)
for i in range(n2):
    for j in range(i+1,n2):
        t = [cols1[i], cols1[j]]
        cols.append(t)

import gc

def writeCols(col):
    colstr = "".join(col)
    print 'writeCols',col
    for i in range(24, 32):
        if os.path.exists('../data/dup/{}{}_ratio.csv'.format(colstr,i)):
            continue
        t = d[(d['day'] < i)&(d['day'] >= i-7)][['label']+col]
        t = t.groupby(col,as_index=False)['label'].agg({colstr+"ratio":np.mean})

        #t.to_csv('../data/dup/{}_day_ratio1.csv'.format(col),index=None) # 只有 day_ratio 的数据
        t.to_csv('../data/dup/{}{}_ratio.csv'.format(colstr,i),index=None) # 有 day_ratio 和 day_Pcount 数据

        del t

"""
pool = Pool(2)
pool.map(writeCols,cols)
pool.close()
pool.join()
"""

train = pd.read_csv('../data/dup/train_.csv')
valid = pd.read_csv('../data/dup/valid_.csv')
test = pd.read_csv('../data/dup/test_.csv')

for col in cols:
    colstr = "".join(col)
    print 'writeCols', col
    ts = []
    for i in range(26, 32):
        t = pd.read_csv('../data/dup/{}{}_ratio.csv'.format(colstr,i))
        t.loc[:, 'day'] = i
        ts.append(t)

    t = pd.concat(ts)
    del ts
    train = train.merge(t,on=col+['day'],how='left')
    valid = valid.merge(t,on=col+['day'],how='left')
    test = test.merge(t, on=col+['day'], how='left')
    del t

print train.shape,valid.shape,test.shape

train.to_csv('../data/dup/train_12.csv',index=None)
valid.to_csv('../data/dup/valid_12.csv',index=None)
test.to_csv('../data/dup/test_12.csv',index=None)







