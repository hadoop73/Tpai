# coding:utf-8

import pandas as pd
import numpy as np

# 在 user_app_actions 中筛选与 train 有交集的 appID
def app_actions_train():
    user_actions = pd.read_csv('../data/dup/user_app_actions.csv')
    train = pd.read_csv('../data/dup/train.csv')

    ad = pd.read_csv('../data/dup/ad.csv')

    d = train.merge(ad,on='creativeID',how='left')
    del train,ad

    appIDs = pd.unique(d['appID'])

    appIDdf = pd.DataFrame({'appID':appIDs})

    print appIDdf.shape
    print user_actions.shape

    user_actions = user_actions.merge(appIDdf,on='appID')

    print user_actions.shape
    user_actions.to_csv('../data/dup/user_app_actions_train.csv',index=None)

# 统计每个 appID 在各个时间段的 action 次数
def app_action_day():
    user_actions = pd.read_csv('../data/dup/user_app_actions_train.csv')
    user_actions['showTime'] = user_actions['installTime'].apply(lambda x:int((x/100)%100))
    for day in [26,27,29,31]:
        d26 = user_actions[user_actions['installTime']<day*10000]

        d26['count'] = 1
        d = d26.groupby(['appID','showTime'],as_index=False)['count'].sum()
        print d.head()
        d.to_csv('../data/dup/app_action{}.csv'.format(day))

user_actions = pd.read_csv('../data/dup/user_app_actions.csv')
userIDs = pd.unique(user_actions['userID'])
user_unique = pd.DataFrame({'userID':userIDs})
user_unique['userActionState'] = 1
print user_unique.shape
print user_unique.head()
user_unique.to_csv('../data/dup/user_action_state.csv',index=None)

#app_action_day()
#print d26.shape
