# coding:utf-8


import pandas as pd
import numpy as np

 
f12 = pd.read_csv('./featuresImportance189.csv')
f17 = pd.read_csv('./featuresImportance196.csv')
f19 = pd.read_csv('./featuresImportance221.csv')
f125 = pd.read_csv('./featuresImportance299.csv')


f = f12.merge(f17,on='feature',suffixes=('1','2'))
f = f.merge(f19,on='feature')
f = f.merge(f125,on='feature',suffixes=('3','4'))


cols = [c for c in f.columns if 'score' in c]

f.loc[:,'sum'] = f[cols].sum(axis=1)
f.sort_values('sum',inplace=True,ascending=False)

print list(f[f['sum']<30]['feature'].values)

print f





