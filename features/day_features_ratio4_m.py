# coding:utf-8


import pandas as pd
import numpy as np


def xx(x):
    x = sorted(x,reverse=True)
    return x[0]*x[1]

def sxx(x):
    x = sorted(x, reverse=True)
    return x[1]

def txx(x):
    x = sorted(x, reverse=True)
    return x[2]

def adxx(x):
    x = sorted(x, reverse=True)
    return (x[0]+x[1]+x[2]+x[3]+x[4])/5

for f in ['train','valid','test']:
    d = pd.read_csv('../data/dup/{}_.csv'.format(f))
    cols = [c for c in d.columns if 'ratio' in c]
    d.loc[:,'max_ratio'] = d[cols].apply(np.max,axis=1)
    d.loc[:, 'mean_ratio'] = d[cols].apply(np.mean,axis=1)
    d.loc[:, 'xx_ratio'] = d[cols].apply(xx,axis=1)
    d.loc[:, 'sxx_ratio'] = d[cols].apply(sxx, axis=1)
    d.loc[:, 'txx_ratio'] = d[cols].apply(txx, axis=1)
    d.loc[:, 'adxx_ratio'] = d[cols].apply(adxx, axis=1)

    #d.loc[:,'gt'] = 1
    #for c in cols:
    #    meanc = d[c].mean()
    #    d.loc[:,'gt'] += d[c].apply(lambda x:1 if x>meanc else 0)


    print d.head()
    d.to_csv('../data/dup/{}_3p_m.csv'.format(f),index=None)
