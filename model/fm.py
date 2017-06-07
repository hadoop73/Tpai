# coding:utf-8


import pandas as pd
import numpy as np
from fastFM import mcmc


train = pd.read_csv('../data/dup/train_xgb11U.csv')
valid = pd.read_csv('../data/dup/valid_xgb11U.csv')
test = pd.read_csv('../data/dup/test_xgb11U.csv')

train.fillna(0,inplace=True)
valid.fillna(0,inplace=True)
test.fillna(0,inplace=True)

train_Y = train['label']
train.drop('label',axis=1,inplace=True)
valid_Y = valid['label']
valid.drop('label',axis=1,inplace=True)
test_Y = test['label']
test.drop('label',axis=1,inplace=True)

fm = mcmc.FMClassification(n_iter=50,random_state=133)

y = fm.fit_predict_proba(train,train_Y,valid)

print y
