# coding:utf-8

import pandas as pd
import zipfile



t = pd.read_csv('../data/dup/test.csv')

p = pd.read_csv('./test.out',header=None)


t['prob'] = p.values

t[['instanceID','prob']].to_csv('./submission.csv',index=None)

with zipfile.ZipFile('submission1024.zip','w') as fout:
  fout.write('submission.csv',compress_type=zipfile.ZIP_DEFLATED)











