# coding:utf-8

import pandas as pd

import xgboost as xgb
#from xgboost import XGBClassifier as xgb
import numpy as np
from sklearn import metrics
import scipy as sp

def logloss(act, pred):
  epsilon = 1e-15
  pred = sp.maximum(epsilon, pred)
  pred = sp.minimum(1-epsilon, pred)
  ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
  ll = ll * -1.0/len(act)
  return ll


train = pd.read_csv('../data/dup/train_fea21.csv')
trainy = train['label']
train.drop('label',axis=1,inplace=True)

valid = pd.read_csv('../data/dup/valid_fea21.csv')
validy = valid['label']
valid.drop('label',axis=1,inplace=True)

def XGBoost_(train=train,y=trainy,test=None,valid=valid,validy=validy,k=0,num_round=3500,
			 gamma=0.02,min_child_weight=1.1,max_depth=8,lamda=20,scale_pos_weight=6,
			 subsamp=0.7,col_bytree=0.7,col_bylevel=0.7,eta=0.01,file="aac"):
    param = {'booster': 'gbtree',
             'objective': 'binary:logistic',
             # 'eval_metric':'auc',
             'scale_pos_weight':int(scale_pos_weight),
             'gamma': gamma,
             'min_child_weight': min_child_weight,
             'max_depth': int(max_depth),
             'lambda': int(lamda),
             'subsample': subsamp,
             'colsample_bytree': col_bytree,
             'colsample_bylevel': col_bylevel,
             'eta': eta,
             'tree_method': 'exact',
             'seed': 0,
             'nthread': 8}
    dtrain = xgb.DMatrix(train, label=y, missing=-10)
    dvalid = xgb.DMatrix(valid, label=validy, missing=-10)
    watchlist = [(dtrain, 'train'),(dvalid, 'eval')]
    # auc = cv_log['test-auc-mean'].max()
    bst = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=50)
    # make prediction
    #dtest = xgb.DMatrix(test, missing=-10)
    #preds = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
    #p = bst.predict(dtrain, ntree_limit=bst.best_ntree_limit)

    scores = bst.predict(dvalid, ntree_limit=bst.best_ntree_limit)

    pro = pd.DataFrame({'prob': scores})
    lgloss = logloss(validy, scores)
    print "logloss", lgloss
    pro.to_csv('../data/res{}.csv'.format(str(lgloss)), index=None)

    fp, tp, thresholds = metrics.roc_curve(validy, scores, pos_label=1)
    auc = metrics.auc(fp, tp)
    print "AUC:{}".format(auc)

    # get feature score
    feature_score = bst.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
    fs = []
    for (key, value) in feature_score:
        fs.append("{0},{1}\n".format(key, value))
    ff = './featuresImportance{0}.csv'.format(lgloss)
    with open(ff, 'w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)
    return -lgloss

#test = pd.read_csv('../data/dup/dtest.csv')
#testy = test['label']
#test.drop('label',axis=1,inplace=True)



from bayes_opt import bayesian_optimization

xgbBo = bayesian_optimization.BayesianOptimization(XGBoost_,{
    "gamma":(0.01,1),
    'min_child_weight':(1,5),
    'max_depth':(5,10),
    'lamda':(5,30),
    'scale_pos_weight':(4,8),
    'subsamp':(0.5,0.9),
    'col_bytree':(0.5,0.9),
    'col_bylevel':(0.5,0.9),
    'eta':(0.01,1)
})

xgbBo.maximize()


