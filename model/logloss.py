# coding:utf-8


import scipy as sp
import pandas as pd
import numpy as np


def logloss(act, pred):
  epsilon = 1e-15
  pred = sp.maximum(epsilon, pred)
  pred = sp.minimum(1-epsilon, pred)
  ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
  ll = ll * -1.0/len(act)
  return ll


valid2 = pd.read_csv('../data/dup/dvalid2.csv')
pred = pd.read_csv('../data/dup/va2.out',header=None)
pred.columns = ['pred']

print logloss(valid2['label'],pred['pred'])
"""

valid2 = pd.read_csv('../data/dup/dtrain1.csv')

print valid2[valid2['label']==0].shape
print valid2[valid2['label']==1].shape
"""