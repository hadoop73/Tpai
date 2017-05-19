#!/home/hadoop/env2.7/bin/python
# coding:utf-8


import argparse
import pandas as pd

"""
    特征产生思路：1) 先用 train21all.csv 产生所有数据的特征
                2) 再用 train21.csv 的数据与 train21all.csv 的特征进行 join 合并
                3) 对 valid test 同样可以通过步骤 2) 获得特征
"""


"""
1 获得 appID 的激活情况
"""

train = pd.read_csv('../data/dup/train21all.csv')
ad = pd.read_csv('../data/dup/ad.csv')
user = pd.read_csv('../data/dup/user.csv')

train = train.merge(ad,on='creativeID',how='left')
train = train.merge(user,on='userID',how='left')

for col in ['creativeID','adID','camgaignID','appID','appPlatform','advertiserID']:
    t = pd.read_csv('../data/ad/train_ad_{0}_{1}.csv'.format(col,'all'))
    train = train.merge(t,on=col,how='left')


for col in ['userID','gender','education','marriageStatus',
            'haveBaby']:
    t = pd.read_csv('../data/user/train_user_{0}_{1}.csv'.format(col,'all'))
    train = train.merge(t,on=col,how='left')

train.to_csv('../data/dup/train21all_ad_user.csv',index=None)
print train.shape
print train.head()

