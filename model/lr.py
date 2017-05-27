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

atrain = pd.read_csv('../data/dup/train_xgb11U.csv'.format(rand))
#atrain = pd.read_csv('../data/dup/train_xgbASample.csv')
atrain.fillna(0,inplace=True)
avalid = pd.read_csv('../data/dup/valid_xgb11U.csv'.format(rand))
test = pd.read_csv('../data/dup/test_xgb11U.csv'.format(rand))

mx = (atrain['age'].max()>avalid['age'].max()) and atrain['age'].max() or avalid['age'].max()

atrain.loc[:,'age'] = atrain['age'].apply(lambda x:1.0*x/mx)
avalid.loc[:,'age'] = avalid['age'].apply(lambda x:1.0*x/mx)
test.loc[:,'age'] = test['age'].apply(lambda x:1.0*x/mx)


trainy = atrain['label']
atrain.drop(['userID','label','creativeID','positionID','adID','appID','appPlatform','clickTime'],axis=1,inplace=True)

avalid.fillna(0,inplace=True)

validy = avalid['label']
avalid.drop(['userID','label','creativeID','positionID','adID','appID','appPlatform','clickTime'],axis=1,inplace=True)
test.fillna(0,inplace=True)
test.drop(['userID','label','creativeID','positionID','adID','appID','appPlatform','clickTime'],axis=1,inplace=True)

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

