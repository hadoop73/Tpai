# coding:utf-8

import xgboost as xgb
import pandas as pd
import numpy as np
import hashlib,math



def hashstr(str, nr_bins=1e+6):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16)%(nr_bins-1)+1


freqfeas = {}
def freqFea(data,cols):
    for col in cols:
        d = data[[col]]
        d['count'] = 1
        d = d.groupby(col,as_index=False)['count'].sum()
        d = d[d['count']>=10]
        freqfeas[col] = set(d[col])

def logFun(x):
    #x = int(1000*x)
    x = int(x)
    if x<2:
        return "sp"+str(x)
    else:
        return str(int(math.log(float(x))**2))

def  dataFFM(f):
    train = pd.read_csv('../data/dup/{}_xgb11U.csv'.format(f))
    train.fillna(0, inplace=True)
    y = train['label']
    train.drop('label', axis=1, inplace=True)
    bst = xgb.Booster({'nthread': 8})
    bst.load_model('../model/bayes/bst150.model')  # 122
    dtrain = xgb.DMatrix(train)
    treeIndex = bst.predict(dtrain, pred_leaf=True)

    df = pd.DataFrame(treeIndex)

    # 类别特征处理
    d = train # 读取原始数据中的 类别特征
    cols = ['adID','camgaignID','appID','appPlatform','advertiserID','creativeID','sitesetID',
            'positionType','positionID','telecomsOperator','connectionType','gender',
            'education','marriagedStatus','haveBaby','hometown','residence','liveState']

    cols = [col for col in cols if col in d.columns]
    freqFea(d,cols)
    nc = d.shape[1]
    columns = [col+"{:05d}".format(j) for j,col in enumerate(d.columns)]
    d.columns = columns
    from multiprocessing import Pool
    pool = Pool(4)

    def delCols(col):
        n = int(col[-5:])
        c = col[:-5]
        if col in cols:
            d.loc[:, col] = d[col].apply(
                lambda x: "{0}:{1}:1".format(n, hashstr(str(x)) if x in freqfeas[c] else hashstr("sparse" + str(x))))
        elif d[col].max() <= 1:
            d.loc[:, col] = d[col].apply(lambda x: "{0}:{1}:1".format(n, logFun(1000 * x)))
        else:
            d.loc[:, col] = d[col].apply(lambda x: "{0}:{1}:1".format(n, logFun(x)))

    pool.map(delCols,columns)
    pool.close()
    pool.join()
    """
    for col in d.columns:
        if col in cols:
            d.loc[:,col] = d[col].apply(lambda x:"{0}:{1}:1".format(n,hashstr(str(x)) if x in freqfeas[col] else hashstr("sparse"+str(x))))
        elif d[col].max()<=1:
            d.loc[:,col] = d[col].apply(lambda x:"{0}:{1}:1".format(n,logFun(1000*x)))
        else:
            d.loc[:, col] = d[col].apply(lambda x: "{0}:{1}:1".format(n, logFun(x)))
    """
    pool = Pool(4)
    columns = [col + "{:05d}".format(j) for j, col in enumerate(df.columns,start=nc)]
    df.columns = columns

    def dfColumns(col):
        n = int(col[-5:])
        df.loc[:, col] = df[col].apply(
            lambda x: "{0}:{1}:1".format(n, hashstr(str(x))))
    pool.map(dfColumns,columns)
    pool.close()
    pool.join()

    """
    for col in df.columns:
        df.loc[:, col] = df[col].apply(
                lambda x: "{0}:{1}:1".format(n, hashstr(str(x))))

    """
    df = d.join(df)
    d = pd.DataFrame({'label':y})
    d = d.join(df)
    print d.head()
    print d.shape
    d.to_csv('../data/dup/{}.ffm'.format(f))


f = 'train'
train = pd.read_csv('../data/dup/{}_xgb11U.csv'.format(f))
train.fillna(0, inplace=True)
y = train['label']
train.drop('label', axis=1, inplace=True)
bst = xgb.Booster({'nthread': 8})
bst.load_model('../model/bayes/bst150.model')  # 122
dtrain = xgb.DMatrix(train)
treeIndex = bst.predict(dtrain, pred_leaf=True)


df = pd.DataFrame(treeIndex)

del dtrain,treeIndex
    # 类别特征处理
d = train  # 读取原始数据中的 类别特征
cols = ['adID', 'camgaignID', 'appID', 'appPlatform', 'advertiserID', 'creativeID', 'sitesetID',
            'positionType', 'positionID', 'telecomsOperator', 'connectionType', 'gender',
            'education', 'marriagedStatus', 'haveBaby', 'hometown', 'residence', 'liveState']

cols = [col for col in cols if col in d.columns]
freqFea(d, cols)
nc = d.shape[1]
columns = [col + "{:05d}".format(j) for j, col in enumerate(d.columns)]
d.columns = columns


def delCols(col):
    n = int(col[-5:])
    c = col[:-5]
    if col in cols:
        d.loc[:, col] = d[col].apply(
            lambda x: "{0}:{1}:1".format(n, hashstr(str(x)) if x in freqfeas[c] else hashstr("sparse" + str(x))))
    elif d[col].max() <= 1:
        d.loc[:, col] = d[col].apply(lambda x: "{0}:{1}:1".format(n, logFun(1000 * x)))
    else:
        d.loc[:, col] = d[col].apply(lambda x: "{0}:{1}:1".format(n, logFun(x)))


def dfColumns(col):
    n = int(col[-5:])
    df.loc[:, col] = df[col].apply(
        lambda x: "{0}:{1}:1".format(n, hashstr(str(x))))

from multiprocessing import Pool

pool = Pool(4)
pool.map(delCols, columns)
pool.close()
pool.join()
"""
    for col in d.columns:
        if col in cols:
            d.loc[:,col] = d[col].apply(lambda x:"{0}:{1}:1".format(n,hashstr(str(x)) if x in freqfeas[col] else hashstr("sparse"+str(x))))
        elif d[col].max()<=1:
            d.loc[:,col] = d[col].apply(lambda x:"{0}:{1}:1".format(n,logFun(1000*x)))
        else:
            d.loc[:, col] = d[col].apply(lambda x: "{0}:{1}:1".format(n, logFun(x)))
"""
pool = Pool(4)
columns = [col + "{:05d}".format(j) for j, col in enumerate(df.columns, start=nc)]
df.columns = columns

pool.map(dfColumns, columns)
pool.close()
pool.join()

"""
    for col in df.columns:
        df.loc[:, col] = df[col].apply(
                lambda x: "{0}:{1}:1".format(n, hashstr(str(x))))

"""

df = d.join(df)
d = pd.DataFrame({'label': y})
d = d.join(df)
print d.head()
print d.shape
d.to_csv('../data/dup/{}.ffm'.format(f))










