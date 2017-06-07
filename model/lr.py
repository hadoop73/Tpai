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

atrain = pd.read_csv('../data/dup/train_3p.csv'.format(rand))
#atrain = pd.read_csv('../data/dup/train_xgbASample.csv')
atrain.fillna(0,inplace=True)
avalid = pd.read_csv('../data/dup/valid_3p.csv'.format(rand))
test = pd.read_csv('../data/dup/test_3p.csv'.format(rand))

cols = ['home', 'hometown', 'residence', 'resid', 'positionID',
        'creativeID', 'adID', 'camgaignID', 'advertiserID', 'appID']

cols2 = ['education', 'haveBaby', 'positionType', 'connectionType']

cols3 = ['gender', 'marriageStatus', 'sitesetID', 'appPlatform', 'telecomsOperator']

cols += cols2 + cols3 + ['label']
cols = [c for c in atrain.columns if c in cols]

trainy = atrain['label']
atrain.drop(cols,axis=1,inplace=True)

avalid.fillna(0,inplace=True)

validy = avalid['label']
avalid.drop(cols,axis=1,inplace=True)
test.fillna(0,inplace=True)
test.drop(cols,axis=1,inplace=True)

lr = LogisticRegression()
lr.fit(atrain,trainy)

prob = lr.predict_proba(avalid)[:,1]

lgloss = logloss(validy, prob)
print "logloss", lgloss


tp = lr.predict_proba(test)[:,1]
t = pd.read_csv('../data/dup/test.csv')

t['prob'] = tp

t[['instanceID','prob']].to_csv('./submission.csv',index=None)

with zipfile.ZipFile('submission1023.zip','w') as fout:
  fout.write('submission.csv',compress_type=zipfile.ZIP_DEFLATED)

