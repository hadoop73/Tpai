# coding:utf-8


import pandas as pd
import numpy as np
import gc

def trainSample(data=None,rate=0.2,rand=133):
    if None == data:
        data = pd.read_csv('../data/dup/train.csv')
    if 'conversionTime' in data.columns:
        data.drop('conversionTime', axis=1, inplace=True)
    valid = data[(data['clickTime'] >= 290000) & (data['clickTime'] < 300000)]
    test = pd.read_csv('../data/dup/test.csv')
    test.drop('instanceID', axis=1, inplace=True)

    data = data[(data['clickTime'] >= 220000) & (data['clickTime'] < 290000)]
    d1 = data[data['label']==1]
    d0 = data[data['label']==0]
    d0 = d0.sample(frac=rate,random_state=rand)
    d1 = d1.sample(frac=rate, random_state=rand)

    data = pd.concat([d0,d1])
    del d0,d1
    gc.collect()
    return data,valid,test

def trainDay(day=29):
    train =  pd.read_csv('../data/dup/train.csv')
    train = train[train['clickTime'] < day*10000]
    train.drop('conversionTime', axis=1, inplace=True)
    return train

def dataSample(data,rate=0.1,rand=133):
    if 'conversionTime' in data.columns:
        data.drop('conversionTime', axis=1, inplace=True)
    valid = data[(data['clickTime'] >= 290000) & (data['clickTime'] < 300000)]
    test = pd.read_csv('../data/dup/test.csv')
    test.drop('instanceID', axis=1, inplace=True)

    d1 = data[data['label'] == 1]
    d0 = data[data['label'] == 0]
    d0 = d0.sample(frac=rate, random_state=rand)
    d1 = d1.sample(frac=rate, random_state=rand)

    data = pd.concat([d0, d1])
    del d0, d1
    gc.collect()
    return data, valid, test

def dataSampleDay(data,rate=0.1,rand=133):
    if 'conversionTime' in data.columns:
        data.drop('conversionTime', axis=1, inplace=True)
    valid = data[(data['clickTime'] >= 29) & (data['clickTime'] < 30)]
    test = data[(data['clickTime']>=31)&(data['clickTime']<32)]

    data = data[(data['clickTime'] >= 22) & (data['clickTime'] < 29)]
    data.to_csv('../data/dup/train_xgbD{}all.csv'.format(rand),index=None)

    d1 = data[data['label'] == 1]
    d0 = data[data['label'] == 0]
    d0 = d0.sample(frac=rate, random_state=rand)
    d1 = d1.sample(frac=rate, random_state=rand)

    data = pd.concat([d0, d1])
    del d0, d1
    gc.collect()
    return data, valid, test


def dataTrain(data,rate=0.1,rand=133):
    if 'conversionTime' in data.columns:
        data.drop('conversionTime', axis=1, inplace=True)
    valid = data[(data['clickTime'] >= 29) & (data['clickTime'] < 30)]
    test = data[(data['clickTime']>=31)&(data['clickTime']<32)]

    data.to_csv('../data/dup/train_xgbD{}all.csv'.format(rand), index=None)
    data = data[(data['clickTime'] >= 27) & (data['clickTime'] < 29)]
    gc.collect()
    return data, valid, test




def dataTrainValid(data,rate=0.1,rand=133):
    if 'conversionTime' in data.columns:
        data.drop('conversionTime', axis=1, inplace=True)
    valid = data[(data['clickTime'] >= 290000) & (data['clickTime'] < 300000)]

    data.to_csv('../data/dup/train_xgbD26{}all.csv'.format(rand), index=None)
    data = data[(data['clickTime'] >= 26) & (data['clickTime'] < 29)]

    d1 = data[data['label'] == 1]
    d0 = data[data['label'] == 0]
    d0 = d0.sample(frac=rate, random_state=rand)
    d1 = d1.sample(frac=rate, random_state=rand)

    data = pd.concat([d0, d1])
    del d0, d1

    gc.collect()
    return data, valid



if __name__=='__main__':
    d = pd.read_csv('../data/dup/train_xgbD26133all.csv')

    valid = d[(d['clickTime'] >= 29) & (d['clickTime'] < 30)]
    valid.to_csv('../data/dup/valid_2.csv',index=None)




