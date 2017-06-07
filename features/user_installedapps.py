# coding:utf-8

import pandas as pd
import numpy as np



user_apps = pd.read_csv('../data/dup/user_installedapps.csv')


f = 'train'

d = pd.read_csv('../data/dup/{}_r.csv'.format(f))

userID = pd.unique(d['userID'])

uId = pd.DataFrame({'userID':userID})
dt = user_apps.merge(uId,on='userID',how='left')

dt.loc[:,'installed'] = 1
d = d.merge(dt,on=['userID','appID'],how='left')


t = dt.groupby('userID',as_index=False)['appID'].agg({'installCount':np.size})

d = d.merge(t,on='userID',how='left')





