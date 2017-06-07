# coding:utf-8



import pandas as pd
import numpy as np


d = pd.DataFrame({'a':[1,2,2],'b':[4,5,6]})

n = d.shape[0]
d.loc[:,'c'] = 0

for i in range(1,n):
    print  d.iloc[i]['a'],d.iloc[i-1]['a']
    d.loc[i,'c'] = (1 if d.iloc[i]['a']==d.iloc[i-1]['a'] else 0)



print d
