#!/home/hadoop/env2.7/bin/python
# coding:utf-8

import pandas as pd

import argparse, csv, sys


parser = argparse.ArgumentParser()
parser.add_argument('csv_path', nargs="*")
args = vars(parser.parse_args())

"""
args = {'csv_path':"../data/dup/dtest.csv",
        'sparse_path':"../data/dup/dtest_sparse.csv"}"""
print args

def getFeatures(file):
    train = pd.read_csv('../data/dup/{}.csv'.format(file))
    train['clickTime'] = train['clickTime'].astype(str).apply(lambda x:int(x[2:4])/4)
    #d = train.drop_duplicates(['clickTime','creativeID','userID','positionID','connectionType','telecomsOperator'])
    d = train
    user = pd.read_csv('../data/dup/user.csv')
    d = d.merge(user,on='userID',how='left')
    d['residence'] = d['residence'].apply(lambda x:x/100)

    cols = ['gender','residence','clickTime','connectionType',
            'telecomsOperator','education','marriageStatus','haveBaby']
    for col in cols:
        dt = d[col]
        t = pd.get_dummies(dt,prefix=col)
        d.drop(col,axis=1,inplace=True)
        d = d.join(t)
    cols = ['creativeID','userID','positionID','hometown']
    if "train" in file:
        cols.append('conversionTime')
    d.drop(cols,axis=1,inplace=True)

    print d.shape
    print d.head()
    d.to_csv("../data/dup/d{}.csv".format(file),index=None)


def noheader(f):
    #d.to_csv('../data/dup/d{}_.csv'.format(f),index=None,header=None,sep=' ')
    data = pd.read_csv("../data/dup/d{}.csv".format(f))
    cols = data.columns
    del data
    with open('../data/dup/d{}_.csv'.format(f), 'w') as fs:
        for row in csv.DictReader(open('../data/dup/d{}.csv'.format(f))):
            feats = []
            for k in cols:
                if k not in ['instanceID','label']:
                    feats.append('{0}'.format(row[k]))
            if int(row['label']) < 0:
                fs.write("0" + ' ' + ' '.join(feats) + '\n')
            else:
                fs.write(row['label'] + ' ' + ' '.join(feats) + '\n')

for f in args['csv_path']:
    d = pd.read_csv("../data/dup/{}.csv".format(f))
    print d.shape
    d.to_csv('../data/dup/{}_.csv'.format(f),sep=' ',header=None,index=None)


def mergeFeatures(file):
    train = pd.read_csv('../data/dup/{}.csv'.format(file))
    print train.shape
    train['clickTime'] = train['clickTime'].astype(str).apply(lambda x:int(x[2:4])/4)
    #d = train.drop_duplicates(['clickTime','creativeID','userID','positionID','connectionType','telecomsOperator'])
    d = train
    user = pd.read_csv('../data/dup/user.csv')
    d = d.merge(user,on='userID',how='left')
    d['residence'] = d['residence'].apply(lambda x:x/100)

    cols = ['creativeID','userID','positionID','hometown']
    if "train" in file:
        cols.append('conversionTime')
    d.drop(cols,axis=1,inplace=True)

    print d.shape
    print d.head()
    d.to_csv("../data/dup/m{}.csv".format(file),index=None)


#for f in ['test','train']:
    #getFeatures(f)
#    noheader(f)
    #mergeFeatures(f)

"""
d = pd.read_csv("../data/dup/dtest_.csv",header=None,sep=' ')
print d.shape

d = pd.read_csv("../data/dup/dtrain_.csv",header=None,sep=' ')
print d.shape
"""



#train = pd.read_csv('../data/dup/train.csv')
#print train.head()
#train.drop(['instanceID'],axis=1,inplace=True)
#train.to_csv("../data/dup/dtest_.csv",header=None)
#getFeatures("test")




