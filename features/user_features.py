# coding:utf-8

import pandas as pd
import numpy as np



train = pd.read_csv('../data/dup/train.csv')
train = train[train['clickTime']<290000]
ad = pd.read_csv('../data/dup/ad.csv')
user = pd.read_csv('../data/dup/user_state.csv')

train = train.merge(ad,on='creativeID',how='left')
train = train.merge(user,on='userID',how='left')

# 用于和 train/valid/test 数据进行 merge
"""
['gender','education','marriageStatus','haveBaby','hometown',
            'residence','zeroSum','liveState','residenceliveState','connectionType',
            'telecomsOperator','clickTime']
"""

train.loc[:,'clickTime'] = train['clickTime'].apply(lambda x:int(x%10000/100))

for col in ['gender','education','marriageStatus','haveBaby','hometown',
            'residence','zeroSum','liveState','residenceliveState','connectionType',
            'telecomsOperator','clickTime']:
    d = train[['label','appID',col]]

    d = d.groupby(['appID',col],as_index=False)['label'].mean()
    d.rename(columns={'label':col+'appIDPositive'},inplace=True)
    print d.head()
    print d.shape
    d.to_csv('../data/user/appID_{}.csv'.format(col),index=None)










