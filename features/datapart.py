# coding:utf-8


import pandas as pd

train = pd.read_csv('../data/dup/train.csv')
train26 = train[train['clickTime']<260000]
train26.to_csv('../data/dup/train26.csv',index=None)



train29No = train[(train['clickTime']>=290000)&(train['label']==0)]
train29No.to_csv('../data/dup/train29No.csv',index=None)






