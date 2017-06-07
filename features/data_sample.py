#!/home/hadoop/env2.7/bin/python
# coding:utf-8


import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

"""
parser = argparse.ArgumentParser()
parser.add_argument('days',type=int,nargs="*")

args = vars(parser.parse_args())

print args
"""
"""
   参数格式：26 27 29 21
   26 27 表示负样本的采样日期
   29 表示验证数据集日期
   21 表示采样数据的后缀，如 train21.csv valid21.csv 为采样训练数据集和验证集
   train21all.csv 表示验证数据集之前的所有数据，也就是 29 日之前的所有数据
"""
train = pd.read_csv('../data/dup/all_sum.csv')
y_train = train['label']
train.drop('label',axis=1,inplace=True)

train,valid,yt,yv = train_test_split(train,y_train,test_size=0.3,random_state=133)


valid.loc[:,'label'] = yv
print 1.0*valid[valid['label']==0].shape[0]/valid[valid['label']==1].shape[0]
valid.to_csv('../data/dup/valid_r2.csv',index=None)

del valid

train,xxx,yt,xxy = train_test_split(train,yt,test_size=0.4,random_state=133)

del xxx

train.loc[:,'label'] = yt
print 1.0*train[train['label']==0].shape[0]/train[train['label']==1].shape[0]

train.to_csv('../data/dup/train_r2.csv',index=None)
#train.to_csv('../data/dup/train_ratio.csv',index=None)
print train.head()
print train.shape