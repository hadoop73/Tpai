# coding:utf-8

import xgboost as xgb
import pandas as pd
import numpy as np
import hashlib,math,gc



def hashstr(str, nr_bins=1e+6):
    return int(int(hashlib.md5(str.encode('utf8')).hexdigest(), 16)%(nr_bins-1)+1)


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
    bst.load_model('../model/bayes/bst138.model')  # 122
    dtrain = xgb.DMatrix(train)
    treeIndex = bst.predict(dtrain, pred_leaf=True)

    df = pd.DataFrame(treeIndex)

    del bst,dtrain,treeIndex
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
        elif d[col].max()<1:
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

def xxffm():
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


def treeFea(f='train'):
    d = pd.read_csv('../data/dup/{}_xgbD133.csv'.format(f))
    d.fillna(0, inplace=True)
    y = d['label']
    d.drop('label', axis=1, inplace=True)
    d.fillna(0,inplace=True)
    bst = xgb.Booster({'nthread': 8})
    bst.load_model('../model/bayes/bst126.model')  # 122
    dtrain = xgb.DMatrix(d)
    tree_leafs = bst.predict(dtrain, pred_leaf=True)

    df = pd.DataFrame(tree_leafs)
    print df.shape
    df.to_csv('../data/dup/{}_treeleaf.csv'.format(f),index=None)
    del d,bst,dtrain,df




def  dataFFMs(f):
    train = pd.read_csv('../data/dup/{}_xgbD133.csv'.format(f))
    train.fillna(0, inplace=True)
    y = train['label']
    train.drop('label', axis=1, inplace=True)



    # 类别特征处理
    d = train # 读取原始数据中的 类别特征
    cols = ['adID','camgaignID','appID','appPlatform','advertiserID','creativeID','sitesetID',
            'positionType','positionID','telecomsOperator','connectionType','gender',
            'education','marriagedStatus','haveBaby','hometown','residence','liveState']

    cols = [col for col in cols if col in d.columns]

    def delDcolumns(c):
        n = int(c[-5:])
        if col in cols:
            d.loc[:, col] = d[col].apply(
                lambda x: "{0}:{1}:1".format(n, hashstr(str(x)) if x in freqfeas[col] else hashstr("sparse" + str(x))))
        elif d[col].max() < 1:
            d.loc[:, col] = d[col].apply(lambda x: "{0}:{1}:1".format(n, logFun(1000 * x)))
        else:
            d.loc[:, col] = d[col].apply(lambda x: "{0}:{1}:1".format(n, logFun(x)))
    freqFea(d,cols)
    nc = d.shape[1]
    d.columns = [col+"{:05d}".format(j) for j,col in enumerate(d.columns)]



    from multiprocessing import Pool
    pool = Pool(4)
    pool.map(delDcolumns,d.columns)
    pool.close()
    pool.join()

    """
    for col in d.columns:
        if col in cols:
            d.loc[:,col] = d[col].apply(lambda x:"{0}:{1}:1".format(n,hashstr(str(x)) if x in freqfeas[col] else hashstr("sparse"+str(x))))
        elif d[col].max()<1:
            d.loc[:,col] = d[col].apply(lambda x:"{0}:{1}:1".format(n,logFun(1000*x)))
        else:
            d.loc[:, col] = d[col].apply(lambda x: "{0}:{1}:1".format(n, logFun(x)))
        n+=1
    """
    df = pd.read_csv('../data/dup/{}_treeleaf.csv'.format(f))
    df.columns = [col + "{:05d}".format(j) for j, col in enumerate(df.columns,start=nc)]

    def delDF(c):
        n = int(c[-5:])
        df.loc[:, col] = df[col].apply(
            lambda x: "{0}:{1}:1".format(n, hashstr("{0}:{1}".format(n, hashstr("{0}:{1}".format(n, x))))))

    pool = Pool(4)
    pool.map(delDF, df.columns)
    pool.close()
    pool.join()
    """
    for col in df.columns:
        df.loc[:, col] = df[col].apply(
                lambda x: "{0}:{1}:1".format(n, hashstr("{0}:{1}".format(n,hashstr("{0}:{1}".format(n,x)))) ))
        n+=1
    """

    df = d.join(df)
    d = pd.DataFrame({'label':y})
    d = d.join(df)
    print d.head()
    print d.shape
    d.to_csv('../data/dup/{}.ffm'.format(f),index=None,header=None,sep=' ')

for f in ['train','valid','test']:
    #treeFea(f)
    train = pd.read_csv('../data/dup/{}_d.csv'.format(f))
    train.fillna(0, inplace=True)
    y = train['label']
    train.drop(['label','userID','clickTime'], axis=1, inplace=True)

    # 类别特征处理
    d = train  # 读取原始数据中的 类别特征
    cols = ['adID', 'home','camgaignID', 'appID', 'appPlatform', 'advertiserID', 'creativeID', 'sitesetID',
            'positionType', 'positionID', 'telecomsOperator', 'connectionType', 'gender', 'resid'
            'education', 'marriagedStatus', 'haveBaby', 'hometown', 'residence', 'liveState']

    cols = [col for col in cols if col in d.columns]

    freqFea(d, cols)
    nc = d.shape[1]
    columns = [col + "{:05d}".format(j) for j, col in enumerate(d.columns)]

    d.columns = columns

    def delDcolumns(c):
        n = int(c[-5:])
        col = c[:-5]
        r = {}
        if col in cols:
            r[c] = list(d[c].apply(
                lambda x: "{0}:{1}:1".format(n, hashstr(str(x)) if x in freqfeas[col] else hashstr("sparse" + str(x)))))
        elif d[c].max() < 1:
            r[c] = list(d[c].apply(lambda x: "{0}:{1}:1".format(n,hashstr(logFun(1000 * x)))))
        else:
            r[c] = list(d[c].apply(lambda x: "{0}:{1}:1".format(n,hashstr(logFun(x)))))
        #print r
        return r

    from multiprocessing import Pool

    pool = Pool(4)
    rs = pool.map(delDcolumns, columns)
    pool.close()
    pool.join()
    del d
    ds = {}
    for item in rs:
        ds.update(item)
    d = pd.DataFrame(ds)
    print d.head()
    del rs
    """
    for col in d.columns:
        if col in cols:
            d.loc[:,col] = d[col].apply(lambda x:"{0}:{1}:1".format(n,hashstr(str(x)) if x in freqfeas[col] else hashstr("sparse"+str(x))))
        elif d[col].max()<1:
            d.loc[:,col] = d[col].apply(lambda x:"{0}:{1}:1".format(n,logFun(1000*x)))
        else:
            d.loc[:, col] = d[col].apply(lambda x: "{0}:{1}:1".format(n, logFun(x)))
        n+=1
    """
    df = pd.read_csv('../data/dup/{}.xgb.csv'.format(f))
    columns = [col + "{:05d}".format(j) for j, col in enumerate(df.columns, start=nc)]

    df.columns = columns

    def delDF(c):
        n = int(c[-5:])
        r = {}
        r[c] = list(df[c].apply(
            lambda x: "{0}:{1}:1".format(n, hashstr("{0}:{1}".format(n, hashstr("{0}:{1}".format(n, x)))))))
        return r

    pool = Pool(4)
    rs = pool.map(delDF, df.columns)
    pool.close()
    pool.join()
    del df
    ds = {}
    for item in rs:
        ds.update(item)
    df = pd.DataFrame(ds)
    del rs
    """
    for col in df.columns:
        df.loc[:, col] = df[col].apply(
                lambda x: "{0}:{1}:1".format(n, hashstr("{0}:{1}".format(n,hashstr("{0}:{1}".format(n,x)))) ))
        n+=1
    """

    df = d.join(df)
    d = pd.DataFrame({'label': y})
    d = d.join(df)
    print d.head()
    print d.shape
    d.to_csv('../data/dup/{}.ffm'.format(f), index=None, header=None, sep=' ')
    del d,df
    gc.collect()