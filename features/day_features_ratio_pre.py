# coding:utf-8


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import os

d = pd.read_csv('../data/dup/all.csv')

d = d[d['day']>=21]

cols = ['positionID','creativeID','hour','age','gender','education','connectionType']

flag = False

def writeCols(day):
    colstr = "".join(cols)
    print 'writeCols',colstr
    if flag and os.path.exists('../data/dup/{}_{}_ratio.csv'.format(colstr,day)):
            return
    t = d[(d['day']<day)&(d['day']>=day-1)][['label']+cols]
    t = t.groupby(cols,as_index=False)['label'].agg({colstr+"day_ratio":np.mean,
                                                     colstr+ "day_Pcount":np.sum})
    t.fillna(0,inplace=True)
    t.loc[:,colstr+"day_Pcount"] = t[colstr+"day_Pcount"]/3
    print t.head()
    t.to_csv('../data/dup/{}_{}_ratio.csv'.format(colstr,day),index=None) # 有 day_ratio 和 day_Pcount 数据


days = range(24,32)
pool = Pool(4)
pool.map(writeCols,days)
pool.close()
pool.join()


rst = []
def delPart(dt):
    day = dt['day'].max()
    print 'time:',day
    colstr = "".join(cols)
    t = pd.read_csv('../data/dup/{}_{}_ratio.csv'.format(colstr,day))
    dt = dt.merge(t,on=cols,how='left')
    dt.fillna(0, inplace=True)
    return dt

dts = [d[d['day']==i] for i in range(26,32)]

pool = Pool(4)
rst = pool.map(delPart,dts)
pool.close()
pool.join()
del dts

train = pd.concat(rst)


test = train.loc[train['day']>=31,:]
test.to_csv('../data/dup/test_.csv',index=None)

print "test size:",test.shape,

train = train.loc[train['day']<30,:]

y_train = train['label']
train.drop('label',axis=1,inplace=True)

train,valid,yt,yv = train_test_split(train,y_train,test_size=0.3,random_state=42)

valid.loc[:,'label'] = yv
valid.to_csv('../data/dup/valid_.csv',index=None)

print 'valid size:',valid.shape,

del valid

train,xxx,yt,xxy = train_test_split(train,yt,test_size=0.6,random_state=42)

del xxx

train.loc[:,'label'] = yt
train.to_csv('../data/dup/train_.csv',index=None)

print 'train size:',train.shape








