# coding:utf-8


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import os

"""
rst = []
for i in range(24,30):
    #t = pd.read_csv('../data/dup/dt{}.csv'.format(i))

    t = pd.read_csv('../data/dup/dt3prior{}_.csv'.format(i)) # 存放有 ratio 和 sum 的数据统计
    #d1 = t[t['label'] == 1]
    #d0 = t[t['label'] == 0]
    #d0 = d0.sample(frac=0.25, random_state=133)
    #d1 = d1.sample(frac=0.25, random_state=133)
    #rst += [d0,d1]
    rst.append(t)

train = pd.concat(rst)

print train.shape
train.to_csv('../data/dup/all24prior_.csv',index=None)
"""
train = pd.read_csv('../data/dup/all24prior_.csv')


y_train = train['label']
train.drop('label',axis=1,inplace=True)

train,valid,yt,yv = train_test_split(train,y_train,test_size=0.75,random_state=42)


train.loc[:,'label'] = yt
print train.shape


#train.to_csv('../data/dup/train_3p_1.csv',index=None)


train,valid,yt,yv = train_test_split(valid,yv,test_size=0.3,random_state=133)

valid.loc[:,'label'] = yv

print valid.shape
valid.to_csv('../data/dup/valid_3p_133.csv',index=None)


