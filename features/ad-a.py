
# coding:utf-8


import pandas as pd
import argparse

#parser = argparse.ArgumentParser()

#parser.add_argument()

train = pd.read_csv('../data/dup/train_fea{}.csv'.format(21))
ntrain = train.shape[0]

valid = pd.read_csv('../data/dup/valid_fea{}.csv'.format(21))
nvalid = valid.shape[0]

d = pd.concat([train,valid])

print "ntrain,nvalid:",ntrain,nvalid

cols = d.columns

cate_cols = []
for col in cols:
    if '_' in col and train[col].sum()>10:
            cate_cols.append(col)
d['label'] = d['label'].astype(str)

train = d.iloc[:ntrain,:]
valid = d.iloc[ntrain:,:]
print d['label'].dtype

del d


with open('../data/dup/train.sparse','w') as ts:
    for i in range(ntrain):
        ones = []
        for j,fea in enumerate(cate_cols,start=1):
            if train.iloc[i][fea] > 0:
                ones.append(str(j))
        ts.write(str(train.iloc[i]['label'])+" "+" ".join(ones)+"\n")

train.drop(cate_cols,axis=1,inplace=True)
train.to_csv("../data/train.dence",index=None,header=None)

with open('../data/dup/valid.sparse','w') as tv:
    for i in range(nvalid):
        ones = []
        for j,fea in enumerate(cate_cols,start=1):
            if valid.iloc[i][fea] > 0:
                ones.append(str(j))
        tv.write(str(valid.iloc[i]['label'])+" "+" ".join(ones)+"\n")

valid.drop(cate_cols,axis=1,inplace=True)
valid.to_csv("../data/valid.dence",index=None,header=None)





