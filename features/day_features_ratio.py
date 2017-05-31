# coding:utf-8


import pandas as pd
import numpy as np


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
d.loc[:,'clickTime'] = d['clickTime'].apply(lambda x:int(x/10000))

d.loc[:,'hometown'] = d['hometown'].apply(lambda x:int(x/100))
d.loc[:,'residence'] = d['residence'].apply(lambda x:int(x/100))

d = d[d['clickTime']>=19]

cols = ['gender','education','marriageStatus','haveBaby','hometown','residence','sitesetID',
        'adID','camgaignID','appID','appPlatform','creativeID','advertiserID','positionID','positionType',
        'connectionType','telecomsOperator']
import gc

#cols = ['gender']
def writeCols(col):
    print 'writeCols',col
    t = d[['label','clickTime',col]]
    t = t.groupby([col,'clickTime'],as_index=False)['label'].agg({col+"day_ratio":np.mean,
                                                                  col + "day_Pcount":np.sum})
    t.to_csv('../data/dup/{}_day_ratio.csv'.format(col),index=None)

    t = d[['label', 'clickTime', 'hour', col]]
    t = t.groupby([col, 'clickTime', 'hour'], as_index=False)['label'].agg({col+"hour_ratio":np.mean,
                                                                    col + "hour_Pcount":np.sum})
    t.to_csv('../data/dup/{}_hour_ratio.csv'.format(col),index=None)


from multiprocessing import Pool
pool = Pool(8)
pool.map(writeCols,cols)
pool.close()
pool.join()

def delPart(dt):
    for col in cols:
        t = pd.read_csv('../data/dup/{}_day_ratio.csv'.format(col))
        for i in range(3):
            t.loc[:,'clickTime'] = t['clickTime'] + 1
            dt = dt.merge(t,on=[col,'clickTime'],how='left')
            dt.rename(columns={col+"day_ratio":col+str(i+1)+"day_ratio"},inplace=True)
            dt.rename(columns={col + "day_Pcount": col + str(i + 1) + "day_Pcount"}, inplace=True)
            gc.collect()

        dt.loc[:,col+"day_2ratio"] = dt[[col+"1day_ratio",col+"2day_ratio"]].apply(np.mean,axis=1)
        dt.loc[:,col+"day_2Pcount"] = dt[[col+"1day_Pcount",col+"2day_Pcount"]].apply(np.mean,axis=1)

        dt.loc[:,col+"day_3ratio"] = dt[[col+"1day_ratio",col+"2day_ratio",col+"3day_ratio"]].apply(np.mean,axis=1)
        dt.loc[:,col+"day_3Pcount"] = dt[[col+"1day_Pcount",col+"2day_Pcount",col+"3day_Pcount"]].apply(np.mean,axis=1)

        t = pd.read_csv('../data/dup/{}_hour_ratio.csv'.format(col))

        for i in range(3):
            t.loc[:, 'clickTime'] = t['clickTime'] + 1
            dt = dt.merge(t, on=[col, 'clickTime', 'hour'], how='left')
            dt.rename(columns={col+"hour_ratio": col + str(i + 1) + "hour_ratio"}, inplace=True)
            dt.rename(columns={col+"hour_Pcount": col + str(i + 1) + "hour_Pcount"}, inplace=True)

            gc.collect()

        dt.loc[:,col+"hour_2ratio"] = dt[[col+"1hour_ratio",col+"2hour_ratio"]].apply(np.mean, axis=1)
        dt.loc[:,col+"hour_2Pcount"] = dt[[col+"1hour_Pcount",col+"2hour_Pcount"]].apply(np.mean, axis=1)

        dt.loc[:,col+"hour_3ratio"] = dt[[col+"1hour_ratio",col+"2hour_ratio",col+"3hour_ratio"]].apply(np.mean,
                                                                                           axis=1)
        dt.loc[:,col+"hour_3Pcount"] = dt[[col+"1hour_Pcount",col+ "2hour_Pcount", col + "3hour_Pcount"]].apply(np.mean,
                                                                                                      axis=1)
    print dt.head()
    return dt

dts = [d[d['clickTime']==i] for i in range(22,32)]

pool = Pool(6)
rst = pool.map(delPart,dts)
pool.close()
pool.join()

d = pd.concat(rst)

from train_sample import dataSampleDay,dataTrain
train,valid,test = dataSampleDay(d,rate=0.17)

print train.head()
print train.shape,valid.shape,test.shape

#train.to_csv('../data/dup/train_xgbD133.csv',index=None)

train.to_csv('../data/dup/train_ratio.csv',index=None)
valid.to_csv('../data/dup/valid_ratio.csv',index=None)
test.to_csv('../data/dup/test_ratio.csv',index=None)


