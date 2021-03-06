# coding:utf-8

import pandas as pd
from sklearn.linear_model import LogisticRegression
import scipy as sp
import zipfile


t = pd.read_csv('../data/dup/test.csv')

pred = pd.read_csv('../data/res.out',header=None)


t['prob'] = pred.values

t[['instanceID','prob']].to_csv('./submission.csv',index=None)

with zipfile.ZipFile('submission101.zip','w') as fout:
  fout.write('submission.csv',compress_type=zipfile.ZIP_DEFLATED)





