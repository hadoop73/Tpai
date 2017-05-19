# coding:utf-8

import pandas as pd

import xgboost as xgb
#from xgboost import XGBClassifier as xgb
import numpy as np
from sklearn import metrics


def XGBoost_(train=None,y=None,test=None,valid=None,validy=None,k=0,num_round=3500,
			 gamma=0.02,min_child_weight=1.1,max_depth=5,lamda=10,scale_pos_weight=3,
			 subsamp=0.7,col_bytree=0.7,col_bylevel=0.7,eta=0.01,file="aac"):
    param = {'booster': 'gbtree',
             'objective': 'binary:logistic',
             # 'eval_metric':'auc',
             'gamma': gamma,
             'min_child_weight': min_child_weight,
             'max_depth': max_depth,
             'lambda': lamda,
             'subsample': subsamp,
             'colsample_bytree': col_bytree,
             'colsample_bylevel': col_bylevel,
             'eta': eta,
             'tree_method': 'exact',
             'seed': 0,
             'nthread': 12}
    dtrain = xgb.DMatrix(train, label=y, missing=-10)
    dvalid = xgb.DMatrix(valid, label=validy, missing=-10)
    watchlist = [(dtrain, 'train')] #, (dvalid, 'eval')]
    # auc = cv_log['test-auc-mean'].max()
    bst = xgb.train(param, dtrain, num_round, watchlist, maximize=True, early_stopping_rounds=50)
    # make prediction
    dtest = xgb.DMatrix(test, missing=-10)
    #preds = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
    #p = bst.predict(dtrain, ntree_limit=bst.best_ntree_limit)

    scores = bst.predict(dvalid, ntree_limit=bst.best_ntree_limit)
    fp, tp, thresholds = metrics.roc_curve(validy, scores, pos_label=1)
    auc = metrics.auc(fp, tp)
    print "AUC:{}".format(auc)

    # get feature score
    feature_score = bst.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
    fs = []
    for (key, value) in feature_score:
        fs.append("{0},{1}\n".format(key, value))
    ff = './featuresImportance.csv'
    with open(ff, 'w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)


train = pd.read_csv('../data/dup/dtrain1.csv')
trainy = train['label']
train.drop('label',axis=1,inplace=True)

valid = pd.read_csv('../data/dup/dvalid2.csv')
validy = valid['label']
valid.drop('label',axis=1,inplace=True)

test = pd.read_csv('../data/dup/dtest.csv')
testy = test['label']
test.drop('label',axis=1,inplace=True)


XGBoost_(train=train,y=trainy,test=test,valid=valid,validy=validy)


