# coding:utf-8

import xgboost as xgb
import pandas as pd
import numpy as np

def predTest():
    test = pd.read_csv('../data/dup/test_fea21.csv')
    dtest = xgb.DMatrix(test,missing=-10)

    bst = xgb.Booster({'nthread': 8})
    bst.load_model('./bst125.model')
    pred = bst.predict(dtest)

    test = pd.read_csv('../data/dup/test.csv')
    test['prob'] = pred

    res = test[['instanceID','prob']]

    res.to_csv('./submission.csv',index=None)

def predIndex(f='train'):
    train = pd.read_csv('../data/dup/{}_xgbA.csv'.format(f))
    train.fillna(0,inplace=True)
    y = train['label']
    train.drop('label',axis=1,inplace=True)
    bst = xgb.Booster({'nthread': 8})
    bst.load_model('./bst117.model') #122
    dtrain = xgb.DMatrix(train)
    treeIndex = bst.predict(dtrain,pred_leaf=True)
    df = pd.DataFrame(treeIndex)
    print df.head()
    #del treeIndex,train,bst

    d = train # 读取原始数据中的 类别特征
    cols = []
    for col in d.columns:
        d.loc[:,col] = pd.cut(d[col],bins=100,labels=range(100))


    df = d.join(df)
    n = df.shape[1]
    df.columns = range(n)

    #df = df.rank(method='dense')
    #print df.head()
    ns = 0 # index
    cols = df.columns
    for col in cols:
        df.loc[:,col] = df[col].rank(method='dense')
        df.loc[:,col] += ns
        ns = pd.unique(df[col]).max()
    #print df.head()
    for col in cols:
        df.loc[:, col] = df[col].apply(lambda x: str(col) + ":" + str(int(x)) + ":" + "1")
    print df.head()
    dd = pd.DataFrame({'label':y})
    dd = dd.join(df)
    #df['label'] = y
    dd.to_csv('../data/dup/{}.ffm'.format(f), index=None, header=None, sep=' ')
    #df.to_csv('../data/dup/{}_cate_gbdt.ffm'.format(f), index=None, header=None, sep=' ')
    #df.to_csv('../data/dup/train_code.csv',index=None)

    #print treeIndex.shape
    #print treeIndex[:5,:]



def part(f='valid'):
    train = pd.read_csv('../data/dup/{}.ffm'.format(f),header=None,sep=' ')
    train0 = train[train[0]==0]
    train1 = train[train[0]==1]
    nn = train1.shape[0]
    train0 = train0.sample(n=nn,random_state=133)
    print train0.shape
    print train1.shape

    train = pd.concat([train0,train1])
    train.to_csv('../data/dup/{}_part.ffm'.format(f), index=None, header=None, sep=' ')
#for f in ['train','valid']:
#    part(f)

def ratio2ffm(f=''):
    train = pd.read_csv('../data/dup/{}_fea21.csv'.format(f))
    cols = train.columns
    ratioCols = ['label']
    for col in cols:
        if 'ratio' in col:
            ratioCols.append(col)

    data_ratio = train[ratioCols]
    n = 64
    idx = 469
    for col in ratioCols[1:]:
        data_ratio.loc[:,col] = data_ratio[col].apply(lambda x:str(n)+":"+str(idx)+":"+str(x))
        n += 1
        idx += 1

    print data_ratio.head()
    data_ratio.to_csv('../data/dup/{}_ratio.csv',index=None)


def joinRatioCate(f):
    dcate = pd.read_csv('../data/dup/{}_cate_gbdt.ffm'.format(f),sep=' ',header=None)
    dratio = pd.read_csv('../data/dup/{}_ratio.csv')
    d = dcate.join(dratio)

    d.to_csv('../data/dup/{}.ffm'.format(f),index=None,header=None,sep=' ')
    print d.head()
    print f, 'size: ', d.shape

for f in ['valid']:#,'test']:
    predIndex(f)
    #ratio2ffm(f)
    #joinRatioCate(f)

def cate2ffm():
    df = pd.read_csv('../data/dup/train_code.csv')
    for col in df.columns:
        df.loc[:, col] = df[col].apply(lambda x:str(int(col) + 1) + ":" + str(int(x)) + ":" + "1")
    print df.head()
    df.to_csv('../data/dup/train_gbdt.ffm', index=None, header=None, sep=' ')

#cate2ffm()



"""
a = pd.DataFrame({'a':[1,2,2,3],'b':[2,7,7,8]})
print a.head()

df = a.rank(method='dense')
ns = pd.unique(df['a']).max()
cols = df.columns[1:]
for col in cols:
    df.loc[:, col] =  df[col].apply(lambda x:x+ns)
    ns = pd.unique(df[col]).max()
print df.head()
"""