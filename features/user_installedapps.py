# coding:utf-8

import pandas as pd
import numpy as np



user_apps = pd.read_csv('../data/dup/user_installedapps.csv')

# userID 的过去安装 app 情况
def user_state():
    users = pd.unique(user_apps['userID'])

    user_apps_state = pd.DataFrame({'userID':users})
    user_apps_state['userAppsState'] = 1

    print user_apps_state.head()
    print user_apps_state.shape

    user_apps_state.to_csv('../data/user_installedapps/user_app_state.csv',index=None)

# app 过去安装次数
def app_install_count():
    train = pd.read_csv('../data/dup/train.csv')
    ad = pd.read_csv('../data/dup/ad.csv')

    train = train.merge(ad,on='creativeID',how='left')

    apps = pd.unique(train['appID'])

    appDf = pd.DataFrame({'appID':apps})

    user_apps = user_apps.merge(appDf,on='appID')
    user_apps['count'] = 1
    d = user_apps.groupby('appID',as_index=False)['count'].agg({'appInstallcount':np.size})

    print d.head()
    d.to_csv('../data/user_installedapps/app_count.csv',index=None)


