#!/home/hadoop/env2.7/bin/python
# coding:utf-8


import pandas as pd
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('days',type=int,nargs="*")

args = vars(parser.parse_args())

print args

"""
   参数格式：26 27 29 21
   26 27 表示负样本的采样日期
   29 表示验证数据集日期
   21 表示采样数据的后缀，如 train21.csv valid21.csv 为采样训练数据集和验证集
   train21all.csv 表示验证数据集之前的所有数据，也就是 29 日之前的所有数据
"""
def GetTrainValidDatas():
    train = pd.read_csv('../data/dup/train.csv')
    # 采样负样本，采样 26 27 日负样本
    train0 = train[train['label']==0]
    d = []
    for day in args['days'][:-2]:
        t = train0[(train0['clickTime']>=day*10000)&(train0['clickTime']<day*10000+10000)]
        d.append(t)
    # 采样正样本,小于 29 日的所有正样本
    t = train[(train['label']==1)&(train['clickTime']<args['days'][-2]*10000)]
    d = pd.concat(d+[t])
    d.to_csv('../data/dup/train{}.csv'.format(args['days'][-1]),index=None)
    print d.head()
    print "train{} size:".format(args['days'][-1]),d.shape
    # 采样验证数据集 valid21.csv
    valid = train[(train['clickTime']>=args['days'][-2]*10000)&(train['clickTime']<args['days'][-2]*10000+10000)]
    valid.to_csv('../data/dup/valid{}.csv'.format(args['days'][-1]),index=None)
    print valid.head()
    print "valid{} size:".format(args['days'][-1]), valid.shape
    # 验证数据集之前的所有数据 train21all.csv
    t = train[(train['clickTime']<args['days'][-2]*10000)]
    t.to_csv('../data/dup/train{}all.csv'.format(args['days'][-1]),index=None)
    print t.head()
    print "train{}all size:".format(args['days'][-1]), t.shape


GetTrainValidDatas()




