# coding:utf-8


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import os

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



#cols1 = ['home', 'resid', 'positionID','hour','age',
#        'creativeID', 'adID', 'camgaignID','advertiserID','appID']

cols1 = ['home', 'resid','haveBaby','marriageStatus','education','gender', 'positionID', 'hour', 'age',
         'creativeID', 'adID', 'camgaignID', 'advertiserID','appPlatform', 'appID','sitesetID','positionType',
         'telecomsOperator','connectionType']

cols = []

n2 = len(cols1)
for i in range(n2):
    for j in range(i+1,n2):
        for k in range(j+1,n2):
            t = [cols1[i], cols1[j],cols1[k]]
            cols.append(t)

flag = True


import gc

def writeCols(col):
    colstr = "".join(col)
    print 'writeCols',col
    for i in range(24, 32):
        if flag and os.path.exists('../data/dup/{}{}_ratio_3.csv'.format(colstr,i)):
            continue
        t = d[(d['day'] < i)&(d['day']>=i-7)][['label']+col]
        t = t.groupby(col,as_index=False)['label'].agg({colstr+"ratio":np.mean})
        #t.to_csv('../data/dup/{}{}_ratio_.csv'.format(colstr,i),index=None) # 有    day_ratio 和 day_Pcount 数据
        t.to_csv('../data/dup/{}{}_ratio_3.csv'.format(colstr,i),index=None) # 有    day_ratio 和 day_Pcount 数据
        del t


pool = Pool(2)
pool.map(writeCols,cols)
pool.close()
pool.join()

flag = True

def delPart(dt):
    day = dt['day'].max()
    print 'time:',day
    if flag and os.path.exists('../data/dup/dt3prior{}3.csv'.format(day)):
        return
    for col in cols:
        colstr = "".join(col)
        t = pd.read_csv('../data/dup/{}{}_ratio_3.csv'.format(colstr,day))
        dt = dt.merge(t,on=col,how='left')
        del t
        dt.fillna(0,inplace=True)
    print dt.head()
    #dt.to_csv('../data/dup/dt{}.csv'.format(day),index=None)
    dt.to_csv('../data/dup/dt3prior{}3.csv'.format(day),index=None)
    del dt
    #return dt

dts = [d[d['day']==i] for i in range(24,32)]

pool = Pool(2)
pool.map(delPart,dts)
pool.close()
pool.join()
del dts

rst = []
for i in range(26,30):
    #t = pd.read_csv('../data/dup/dt{}.csv'.format(i))

    t = pd.read_csv('../data/dup/dt3prior{}3.csv'.format(i)) # 存放有 ratio 和 sum 的数据统计
    #d1 = t[t['label'] == 1]
    #d0 = t[t['label'] == 0]
    #d0 = d0.sample(frac=0.25, random_state=133)
    #d1 = d1.sample(frac=0.25, random_state=133)
    #rst += [d0,d1]
    rst.append(t)

train = pd.concat(rst)

print train.shape
train.to_csv('../data/dup/all26prior3.csv',index=None)
#train.to_csv('../data/dup/all26prior.csv',index=None)



print 1.0*train[train['label']==0].shape[0]/train[train['label']==1].shape[0]

y_train = train['label']
train.drop('label',axis=1,inplace=True)

train,valid,yt,yv = train_test_split(train,y_train,test_size=0.3,random_state=42)


valid.loc[:,'label'] = yv
print 1.0*valid[valid['label']==0].shape[0]/valid[valid['label']==1].shape[0]
valid.to_csv('../data/dup/valid_3p3.csv',index=None)

del valid

train,xxx,yt,xxy = train_test_split(train,yt,test_size=0.6,random_state=42)

del xxx

train.loc[:,'label'] = yt
print 1.0*train[train['label']==0].shape[0]/train[train['label']==1].shape[0]

train.to_csv('../data/dup/train_3p3.csv',index=None)
#train.to_csv('../data/dup/train_ratio.csv',index=None)
print train.head()
print train.shape

del train,d
gc.collect()


#train,valid,test = dataSampleDay(d,rate=0.17)
test = pd.read_csv('../data/dup/dt3prior{}3.csv'.format(31))

test.to_csv('../data/dup/test_3p3.csv',index=None)

del test

"""
1, 没有去掉正样本数的统计，生成dt{}.csv,生成训练数据train_ratio.csv
2, 去掉正样本数2day_Pcount的统计，生成 dt1{}.csv，并生成 train_r.csv
"""

#train.to_csv('../data/dup/train_xgbD133.csv',index=None)





