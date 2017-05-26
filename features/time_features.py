# coding:utf-8

import pandas as pd
import numpy as np


train = pd.read_csv('../data/dup/train.csv')
train.drop('conversionTime',axis=1,inplace=True)

ad = pd.read_csv('../data/dup/ad.csv')
d = train.merge(ad,on='creativeID',how='left')

user = pd.read_csv('../data/dup/user_state.csv')
d = d.merge(user,on='userID',how='left')

position = pd.read_csv('../data/dup/position.csv')
d = d.merge(position,on='positionID',how='left')
del train,ad,user,position

#train = train[train['clickTime']<300000]

cols = ['gender', 'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence', 'sitesetID',
        'adID', 'camgaignID', 'appID', 'appPlatform', 'creativeID', 'advertiserID', 'positionID', 'positionType',
        'connectionType', 'telecomsOperator', 'liveState']

def hoursAgo(d=d):
    #  统计前一个小时的浏览次数
    d.loc[:,'clickTime'] = d['clickTime'].apply(lambda x:24*int(x/10000-17)+int(x%10000/100))

    #print d.tail()

    # 统计 26 27 29 31 日之前一个小时的浏览量
    # 也就是时间在 (25-17)*24+23~(27-17)*24+23
    # (28-17)*24+23~(29-17)*24+23  (30-17)*24+23~(31-17)*24+23
    # 统计过去 5 个小时的浏览量
    d = d[((d['clickTime']>=91)&(d['clickTime']<264))|
              ((d['clickTime']>=283)&(d['clickTime']<312))|
              (d['clickTime'] >= 331)]
    """
    d['count'] = 1
    d = d.groupby(['appID','clickTime'],as_index=False)['count'].sum()
    print d.head()
    print d.shape
    d.to_csv('../data/time/onehour.csv',index=None)
    """
    for col in cols:
        t = d.groupby([col, 'clickTime'], as_index=False)['label'].agg({col+'_size_hour': np.size})
        print t.head()
        t.to_csv('../data/time/onehour_{}.csv'.format(col), index=None)


# 同一时间点的前几天的情况，浏览次数，激活次数

def days(d=d):
    d.loc[:,'clickTime'] = d['clickTime'].apply(lambda x:int(x/100))
    d = d.loc[(d['clickTime']>=1700),:]

    for col in cols:
        t = d.groupby([col, 'clickTime'], as_index=False)['label'].agg({col + '_size_day': np.size,
                                                                        col + '_ratio_day': np.mean,
                                                                        col + '_sum_day': np.sum})
        print t.head()
        t.to_csv('../data/time/days_{}.csv'.format(col), index=None)


#hoursAgo(d)
days(d)
#activate(d)