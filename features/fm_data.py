# coding:utf-8

import pandas as pd
import xgboost as xgb
import re,gc
from sklearn.utils import shuffle

freqfeas = {}
def freqFea(data,cols):
    for col in cols:
        d = data[col]
        d['count'] = 1
        d = d.groupby(col,as_index=False)['count'].sum()
        d = d[d['count']>=10]
        freqfeas[col] = set(d[col])

def fmDatas(f):
    train = pd.read_csv('../data/dup/{}_xgb11U.csv'.format(f))
    train.fillna(0, inplace=True)
    y = train['label']
    train.drop('label', axis=1, inplace=True)

    bst = xgb.Booster({'nthread': 4})
    bst.load_model('../model/bayes/bst118.model')  # 122
    dtrain = xgb.DMatrix(train)
    treeIndex = bst.predict(dtrain, pred_leaf=True)
    print treeIndex.head()
    del dtrain,train
    df = pd.DataFrame(treeIndex)

    columns = df.columns
    df = df.rank(method='dense')
    n = 275
    nl = df.shape[0]
    for col in columns:
        df.loc[:,col] = df[col].apply(lambda x:"{0}:1".format(n+int(x)))
        n = n + int(df[col].max()) + 1
    print df.head()
    df.to_csv('../data/dup/{}.tmp.xgb.fm'.format(f),index=None,header=None,sep=' ')

    p = re.compile('\s+')
    with open('../data/dup/{}.tmp.dense.fm',mode='r') as fd, \
        open('../data/dup/{}.tmp.xgb.fm'.format(f),mode='r') as fx, \
        open('../data/dup/{}.fm'.format(f),mode='w') as fw:
        for i in range(nl):
            sd = fd.readline()
            ssd = p.sub(' ',sd)
            sx = fx.readline()
            ssx = p.sub(' ',sx)

            fw.write(ssd.strip()+' '+ssx.strip()+'\n')


#fmDatas('train')

def fmdt(f='train'):
    train = pd.read_csv('../data/dup/{}_xgbD133.csv'.format(f))
    train.fillna(0, inplace=True)
    y = train['label']
    train.drop('label', axis=1, inplace=True)

    # 类别特征处理
    cols = ['adID', 'camgaignID', 'appID', 'appPlatform', 'advertiserID', 'creativeID', 'sitesetID',
            'positionType', 'positionID', 'telecomsOperator', 'connectionType', 'gender', 'userID',
            'education', 'marriagedStatus', 'haveBaby', 'hometown', 'residence', 'liveState']
    cols = [col for col in cols if col in train.columns]
    # for col in cols:
    #    t = pd.get_dummies(d[col], prefix=col)
    #    d = d.join(t)
    train.drop(cols, axis=1, inplace=True)
    n = 0
    for col in train.columns:
        train.loc[:, col] = train[col].apply(lambda x: '{0}:{1}'.format(n, x) if x != None and x != 0 else "")
        n+=1

    tmp = pd.DataFrame({'label': y})
    d = tmp.join(train)
    print d.head()
    #d = shuffle(d)
    #print d.head()
    #d.to_csv('../data/dup/{}.tmp.dense.fm'.format(f), index=None, header=None, sep=' ')

    d.to_csv('../data/dup/{}.tmp_rand.dense.fm'.format(f), index=None, header=None, sep=' ')

"""
for f in ['valid','train','test']:
    #fmdt(f)
    d = pd.read_csv('../data/dup/{}.tmp.dense.fm'.format(f),header=None,sep=' ')
    print d.head()
    d = shuffle(d)
    print d.head()
    d.to_csv('../data/dup/{}.tmp_rand.fm'.format(f), index=None, header=None, sep=' ')
"""
def fmFormate():
    p = re.compile('\s+')
    for f in ['train','valid','test']:
        with open('../data/dup/{}.tmp_rand.dense.fm'.format(f), mode='r') as fd, \
                    open('../data/dup/{}.rand.fm'.format(f), mode='w') as fx:
                for sd in fd.readlines():
                    #sd = fd.readline()
                    ssd = p.sub(' ', sd)
                    #sx = fx.readline()
                    #ssx = p.sub(' ', sx)

                    fx.write(ssd.strip()+'\n')


def xx():
    import gc
    for f in ['train','valid','test']:
        #fmdt(f)
        train = pd.read_csv('../data/dup/{}_xgbD133.csv'.format(f))
        train.fillna(0, inplace=True)
        y = train['label']
        train.drop('label', axis=1, inplace=True)

        # 类别特征处理
        cols = ['adID', 'camgaignID', 'appID', 'appPlatform', 'advertiserID', 'creativeID', 'sitesetID',
                'positionType', 'positionID', 'telecomsOperator', 'connectionType', 'gender', 'userID',
                'education', 'marriagedStatus', 'haveBaby', 'hometown', 'residence', 'liveState']
        cols = [col for col in cols if col in train.columns]
        # for col in cols:
        #    t = pd.get_dummies(d[col], prefix=col)
        #    d = d.join(t)
        train.drop(cols, axis=1, inplace=True)
        train.columns = range(train.shape[1])

        def funCol(col):
            r = {}
            r[col] = train[col].apply(lambda x: '{0}:{1}'.format(col, x) if x != None and x != 0 else "")
            return r


        from multiprocessing import Pool
        pool = Pool(4)
        rs = pool.map(funCol,train.columns)
        #del train
        ds = {}
        for r in rs:
            ds.update(r)
        del rs
        train = pd.DataFrame(ds)
        tmp = pd.DataFrame({'label': y})
        d = tmp.join(train)
        print d.head()
        #d = shuffle(d)
        #print d.head()
        #d.to_csv('../data/dup/{}.tmp.dense.fm'.format(f), index=None, header=None, sep=' ')
        d.to_csv('../data/dup/{}.tmp_rand.dense.fm'.format(f), index=None, header=None, sep=' ')
        #del train,ds,d
        gc.collect()


fmFormate()





