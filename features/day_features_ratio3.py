import pandas as pd
import numpy as np


tall = pd.read_csv('../data/dup/all.csv')

cols = [c for c in tall.columns if 'ratio' in c]


for f in ['train','valid','test']:
    train = pd.read_csv('../data/dup/{}_r.csv'.format(f))

    #print tall.head()
    def xx(x):
        r = 1.0
        for xi in x:
            r *= (xi+1)
        return r

    train.loc[:,'allP'] = tall[cols].apply(xx,axis=1)

    train.to_csv('../data/dup/{}_r2.csv'.format(f),index=None)

