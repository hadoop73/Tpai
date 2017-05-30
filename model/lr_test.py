# coding:utf-8

import pandas as pd
from sklearn.linear_model import LogisticRegression
import scipy as sp
import zipfile

def logloss(act, pred):
  epsilon = 1e-15
  pred = sp.maximum(epsilon, pred)
  pred = sp.minimum(1-epsilon, pred)
  ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
  ll = ll * -1.0/len(act)
  return ll

rand = 1023

atrain = pd.read_csv('../data/dup/train_xgbD133.csv'.format(rand))
#atrain = pd.read_csv('../data/dup/train_xgbASample.csv')
atrain.fillna(0,inplace=True)
avalid = pd.read_csv('../data/dup/valid_xgbD133.csv'.format(rand))


trainy = atrain['label']

avalid.fillna(0,inplace=True)

validy = avalid['label']

cols = ['adID', 'camgaignID', 'appID', 'appPlatform', 'advertiserID', 'sitesetID',
        'positionType', 'positionID', 'telecomsOperator', 'connectionType', 'gender',
        'education', 'marriagedStatus', 'haveBaby', 'liveState']
import gc

dcols = avalid.columns
rcol = ['userID', 'creativeID', 'label', 'camgaignID', 'positionID','marriageStatus1day_hour_ratio',
        'camgaignID3day_hour_ratio','advertiserID1day_ratio','positionType3day_ratio']
for c in dcols:
    if (c  in cols) or (c  in rcol):
            continue
    cs = rcol
    dv = avalid.drop(cs,axis=1)
    dt = atrain.drop(cs, axis=1)

    lr = LogisticRegression()
    lr.fit(dt,trainy)

    prob = lr.predict_proba(dv)[:,1]

    lgloss = logloss(validy, prob)
    print cs,"logloss", lgloss
    gc.collect()





