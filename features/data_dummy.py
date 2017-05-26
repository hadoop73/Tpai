# coding:utf-8


import pandas as pd
import numpy as np

def dataCate():
    data = pd.read_csv('../data/dup/train_fea22.csv')
    ffmcols = []

    for col in data.columns:
        if col not in ['userID'] and '_' not in col:
            ffmcols.append(col)
    d = data[ffmcols]
    d.to_csv('../data/dup/train_cate.csv',index=None)

    dropcols = ['userID', 'gender', 'education', 'marriageStatus', 'positionID', 'hometown',
            'haveBaby', 'creativeID', 'adID', 'camgaignID', 'appID', 'appPlatform', 'advertiserID', 'residence',
            'connectionType', 'telecomsOperator']
    data.drop(dropcols,axis=1,inplace=True)
    data.to_csv('../data/dup/train_xgb.csv',index=None)

    print d.head()

    data = pd.read_csv('../data/dup/valid_fea22.csv')

    d = data[ffmcols]
    d.to_csv('../data/dup/valid_cate.csv',index=None)  # 用于 ffm 的数据
    data.drop(dropcols, axis=1, inplace=True)
    data.to_csv('../data/dup/valid_xgb.csv', index=None) # 用于 xgb 训练
    print d.head()

    data = pd.read_csv('../data/dup/test_fea22.csv')
    d = data[ffmcols]
    d.to_csv('../data/dup/test_cate.csv',index=None)
    data.drop(dropcols, axis=1, inplace=True)
    data.to_csv('../data/dup/test_xgb.csv', index=None)  # 用于 xgb 训练
    print d.head()

"""
  对 train，valid，test 的数据进行类别特征 one hot 处理
"""
def trainAndValid():

    data = pd.read_csv('../data/dup/data_user_hd_act.csv')
    #data = pd.read_csv('../data/dup/train21test_ad_user.csv')
    for col in ['gender','education','marriageStatus','connectionType','telecomsOperator',
                'haveBaby','hometown','residence','appID','appPlatform','advertiserID']:
        t = pd.get_dummies(data[col],prefix=col)
        data = data.join(t)
        data.drop(col, axis=1, inplace=True)

    # 数据离散化
    #data.drop('clickTime', axis=1, inplace=True)
    data.fillna(-10, inplace=True)
    """
    for col in data.columns:
        if col not in ['userID','clickTime','label'] and '_' not in col:
            n = len(pd.unique(data[col]))
            if n > 5:
                n = 5
            data.loc[:, col] = pd.cut(data[col], bins=n, labels=range(n))
            #data[col] = data[col].astype(int)
    """
    # 采样 26 27 日样本，作为训练数据
    train = data[(data['clickTimecp'] >= 240000) & (data['clickTimecp'] < 280000)]
    valid = data[(data['clickTimecp'] >= 290000) & (data['clickTimecp'] < 300000)]
    test = data[(data['clickTimecp'] >= 310000)]

    train.to_csv('../data/dup/train_xgb.csv', index=None)
    valid.to_csv('../data/dup/valid_xgb.csv', index=None)
    test.to_csv('../data/dup/test_xgb.csv', index=None)

#trainAndValid()
#dataCate()


# appID，creativeID 在每天的 每个小时的浏览记录情况,此时统计前一小时的情况
def oneHourAgo():
    train = pd.read_csv('../data/dup/train_fea22.csv')
    ntrain = train.shape[0]
    valid = pd.read_csv('../data/dup/valid_fea22.csv')
    nvalid = valid.shape[0]
    test = pd.read_csv('../data/dup/test_fea22.csv')

    data = pd.concat([train,valid,test])
    del train,valid,test

    for day in [26,27,29,31]:
        t = data[(data['clickTime']>=day*10000)&(data['clickTime']<day*10000+10000)]
        t['showTime'] = t['clickTime'].apply(lambda x:int((x%10000)/100))

        d = t[['creativeID','showTime']]
        d['count'] = 1
        d.fillna(0,inplace=True)
        d['showTime'] = d['showTime'] - 1
        td = d.groupby(['creativeID','showTime'],as_index=False)['count'].sum()
        td.fillna(0,inplace=True)
        print td.head()
        #td = pd.pivot_table(d,index='appID',columns='showTime',values='count',aggfunc=np.sum)
        #td.reset_index(inplace=True)
        #t = t.merge(td,on=['creativeID','showTime'],how='left')
        td.to_csv('../data/dup/one_hour_ago_cID{}.csv'.format(day),index=None)

trainAndValid()
#d = pd.read_csv('../data/dup/one_hour_ago{}.csv'.format(26))
#d.loc[:,'oneHourAgo'] = None

