# coding:utf-8


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

train = pd.read_csv('../data/pre/train.csv')
train.drop(['conversionTime'],axis=1,inplace=True)

test = pd.read_csv('../data/dup/test.csv')
test.drop('instanceID',axis=1,inplace=True)

d = pd.concat([train,test])
del train,test

ad = pd.read_csv('../data/dup/ad.csv')
user = pd.read_csv('../data/dup/user.csv')
position = pd.read_csv('../data/dup/position.csv')
d = d.merge(ad,on='creativeID',how='left')
del ad
d = d.merge(user,on='userID',how='left')
del user
d = d.merge(position,on='positionID',how='left')
del position

d.loc[:,'hour'] = d['clickTime'].apply(lambda x:int(x%10000/100))
# 按天统计
d.loc[:,'day'] = d['clickTime'].apply(lambda x:int(x/10000))

d.loc[:,'home'] = d['hometown'].apply(lambda x:int(x/100))
d.loc[:,'resid'] = d['residence'].apply(lambda x:int(x/100))

d = d[d['day']>=21]

cols = [['home'],['hometown'],['residence'],['resid'],['positionID'],
        ['creativeID'],['adID'],['camgaignID'],['advertiserID'],['appID']]

cols2 = ['education','haveBaby','positionType','connectionType']

cols3 = [('gender',3),('marriageStatus',4),('sitesetID',3),('appPlatform',2),('telecomsOperator',4)]

import gc

#cols = ['gender']
def writeCols(col):
    colstr = "".join(col)
    print 'writeCols',col
    t = d[['label','day']+col]
    t = t.groupby(col+['day'],as_index=False)['label'].agg({colstr+"day_ratio":np.mean,
                                                                  colstr + "day_Pcount":np.sum})
    #t.to_csv('../data/dup/{}_day_ratio1.csv'.format(col),index=None) # 只有 day_ratio 的数据
    t.to_csv('../data/dup/{}_day_ratio.csv'.format(colstr),index=None) # 有 day_ratio 和 day_Pcount 数据

    t = d[['label', 'day', 'hour'] + col]
    t = t.groupby(col+['day', 'hour'], as_index=False)['label'].agg({colstr+"hour_ratio":np.mean,
                                                                           colstr + "hour_Pcount":np.sum})
    t.to_csv('../data/dup/{}_hour_ratio.csv'.format(colstr),index=None)


n2 = len(cols2)
for i in range(n2):
    for j in range(i+1,n2):
        t = [cols2[i], cols2[j]]
        cols.append(t)

n3 = len(cols3)
for i in range(n3):
    for j in range(i+1,n3):
        for k in range(j+1,n3):
            if cols3[i][1]*cols3[j][1]*cols3[k][1] > 20:
                t = [cols3[i][0],cols3[j][0],cols3[k][0]]
                cols.append(t)


print cols

from multiprocessing import Pool

"""
pool = Pool(8)
pool.map(writeCols,cols)
pool.close()
pool.join()
"""


def delPart(dt):
    day = dt['day'].max()
    print 'time:',day
    for col in cols:
        colstr = "".join(col)
        t = pd.read_csv('../data/dup/{}_day_ratio.csv'.format(colstr))

        #t = pd.read_csv('../data/dup/{}_day_ratio1.csv'.format(col))
        t = t[(t['day'] >= day - 3) & (t['day'] < day)]
        ts = []
        for i in range(3):
            t.loc[:,'day'] = t['day'] + 1
            ts.append(t.copy())
        t = pd.concat(ts)
        del ts
        t = t.groupby(col+['day'],as_index=False)[colstr+"day_ratio",colstr + "day_Pcount"].mean()
        dt = dt.merge(t, on=col+['day'], how='left')

        t = pd.read_csv('../data/dup/{}_hour_ratio.csv'.format(colstr))
        t = t[(t['day']>=day-3)&(t['day']<day)]
        ts = []
        for i in range(3):
            t.loc[:, 'day'] = t['day'] + 1
            ts.append(t.copy())

        t = pd.concat(ts)
        del ts
        t = t.groupby(col+['day','hour'], as_index=False)[colstr + "hour_ratio", colstr + "hour_Pcount"].mean()
        dt = dt.merge(t, on=col+['day', 'hour'], how='left')

        #dt.loc[:,col+"hour_2Pcount"] = dt[[col+"1hour_Pcount",col+"2hour_Pcount"]].apply(np.mean, axis=1)
        #dt.loc[:,col+"hour_3Pcount"] = dt[[col+"1hour_Pcount",col+ "2hour_Pcount", col + "3hour_Pcount"]].apply(np.mean,
                                                                                                     # axis=1)
    print dt.head()
    #dt.to_csv('../data/dup/dt{}.csv'.format(day),index=None)

    dt.to_csv('../data/dup/dt1{}.csv'.format(day),index=None)
    #return dt

dts = [d[d['day']==i] for i in range(22,32)]

pool = Pool(6)
pool.map(delPart,dts)
pool.close()
pool.join()
del dts

rst = []
for i in range(26,30):
    #t = pd.read_csv('../data/dup/dt{}.csv'.format(i))

    t = pd.read_csv('../data/dup/dt1{}.csv'.format(i)) # 存放有 ratio 和 sum 的数据统计
    #d1 = t[t['label'] == 1]
    #d0 = t[t['label'] == 0]
    #d0 = d0.sample(frac=0.25, random_state=133)
    #d1 = d1.sample(frac=0.25, random_state=133)
    #rst += [d0,d1]
    rst.append(t)

train = pd.concat(rst)

print train.shape
train.to_csv('../data/dup/all.csv',index=None)



print 1.0*train[train['label']==0].shape[0]/train[train['label']==1].shape[0]

y_train = train['label']
train.drop('label',axis=1,inplace=True)

train,valid,yt,yv = train_test_split(train,y_train,test_size=0.3,random_state=42)


valid.loc[:,'label'] = yv
print 1.0*valid[valid['label']==0].shape[0]/valid[valid['label']==1].shape[0]
valid.to_csv('../data/dup/valid_r.csv',index=None)

del valid

train,xxx,yt,xxy = train_test_split(train,yt,test_size=0.4,random_state=42)

del xxx

train.loc[:,'label'] = yt
print 1.0*train[train['label']==0].shape[0]/train[train['label']==1].shape[0]

train.to_csv('../data/dup/train_r.csv',index=None)
#train.to_csv('../data/dup/train_ratio.csv',index=None)
print train.head()
print train.shape

del train,d
gc.collect()


from train_sample import dataSampleDay,dataTrain
#train,valid,test = dataSampleDay(d,rate=0.17)
test = pd.read_csv('../data/dup/dt1{}.csv'.format(31))

test.to_csv('../data/dup/test_r.csv',index=None)

del test

"""
1, 没有去掉正样本数的统计，生成dt{}.csv,生成训练数据train_ratio.csv
2, 去掉正样本数2day_Pcount的统计，生成 dt1{}.csv，并生成 train_r.csv
"""

#train.to_csv('../data/dup/train_xgbD133.csv',index=None)





