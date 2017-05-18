# coding:utf-8
import numpy as np
import pandas as pd


def dropDup(file):
    data = pd.read_csv("../data/pre/{}.csv".format(file))
    data.drop_duplicates(inplace=True)
    data.to_csv("../data/dup/{}.csv",index=None)


def main():
    for f in ['train','user_app_actions','user_installedapps',
              'test','app_categories','position','user','ad']:
        dropDup(f)

#  数据去重

def ratioPN():
    train = pd.read_csv("../data/pre/train.csv")
    train.head()
    s0 = train[train['label']==1].shape[0]
    s1 = train[train['label']==0].shape[0]
    print s1,s0
    print s1/s0
    """
负样本   正样本  相除结果
3656266 93262
39
"""


train = pd.read_csv("../data/dup/train.csv")
print train.shape

train = train[(train['label']==0)&(train['clickTime']<260000)]

print train.shape
train.to_csv("../data/dup/train_time.csv", index=None)

"""
trainDup()
user_app()
user_installedapps()
test()
app_cate()
position()
user()
ad()
"""

