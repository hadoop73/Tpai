# coding:utf-8


import pandas as pd
import numpy as np


def xx(x):
    r = 1.0
    for xi in x:
        r *= (xi+1)
    return r

for f in ['train','valid','test']:
    d = pd.read_csv('../data/dup/{}_3p.csv'.format(f))

    cols = [c for c in d.columns if 'ratio' in c]
    d.loc[:,'maxratio'] = d[cols].max(axis=1)
    d.loc[:,'minratio'] = d[cols].min(axis=1)
    d.loc[:, 'meanratio'] = d[cols].mean(axis=1)
    d.loc[:, 'stdratio'] = d[cols].std(axis=1)
    d.loc[:, 'xxratio'] = d[cols].apply(xx,axis=1)

    print d.head()
    d.to_csv('../data/dup/{}_m3.csv'.format(f),index=None)




