# coding:utf-8

import pandas as pd

import xgboost as xgb
# from xgboost import XGBClassifier as xgb
import numpy as np
from sklearn import metrics
import scipy as sp


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    return ll


cols = ['home', 'hometown', 'residence', 'resid', 'positionID',
        'creativeID', 'adID', 'camgaignID', 'advertiserID', 'appID']

cols2 = ['education', 'haveBaby', 'positionType', 'connectionType']

cols3 = ['gender', 'marriageStatus', 'sitesetID', 'appPlatform', 'telecomsOperator']

# cols += cols2 + cols3 + ['label']
"""
appID,7
camgaignIDappIDratio,6
homecamgaignIDratio,6
creativeIDadIDratio,6
creativeIDcamgaignIDratio,5
marriageStatus,5
telecomsOperator,4
advertiserIDappIDratio,3
haveBaby,3
adIDadvertiserIDratio,3
sitesetID,2
creativeIDappIDratio,2
appPlatform,1
creativeIDadvertiserIDratio,1
adIDappIDratio,1
positionType,1
"""
# cols = ['positionType','adIDappIDratio','creativeIDadvertiserIDratio','appPlatform',
#        'creativeIDappIDratio','sitesetID','adIDadvertiserIDratio','haveBaby','advertiserIDappIDratio',
#        'haveBaby','advertiserIDappIDratio','camgaignIDadvertiserIDratio','camgaignIDappIDratio1','creativeIDappIDratio1',
#        'camgaignIDappIDratio','telecomsOperator','adIDcamgaignIDratio','connectionTypeadIDratio1','appID',
#        'marriageStatus','creativeIDadvertiserIDratio1','camgaignIDadvertiserIDratio1']


# cols = ['home', 'resid','haveBaby','marriageStatus','education','gender', 'positionID', 'hour', 'age',
#         'creativeID', 'adID', 'camgaignID', 'advertiserID', 'appID','sitesetID','positionType',
#         'telecomsOperator','connectionType']
# cols += ['homeadIDratio', 'houradIDratio', 'residadIDratio', 'residcamgaignIDratio', 'adIDcamgaignIDratio', 'camgaignIDappIDratio', 'adIDadvertiserIDratio', 'creativeIDcamgaignIDratio', 'camgaignIDadvertiserIDratio', 'creativeIDadvertiserIDratio', 'creativeIDappIDratio', 'adIDappIDratio', 'appPlatform']




# cols = ['advertiserIDappIDratio', 'creativeIDcamgaignIDratio', 'creativeID', 'education', 'resid', 'creativeIDadvertiserIDratio', 'haveBaby', 'sitesetID', 'home', 'camgaignIDappIDratio', 'telecomsOperator', 'marriageStatus']



train = pd.read_csv('../data/dup/train_3p_m.csv')
clmns = train.columns
cols = []

print cols
# cols = [c for c in train.columns[:200]]
# cols = [c for c in train.columns if c in cols]

trainy = train['label']
train.drop(cols + ['label'], axis=1, inplace=True)
train.fillna(0, inplace=True)

valid = pd.read_csv('../data/dup/valid_3p_m.csv')
validy = valid['label']
valid.drop(cols + ['label'], axis=1, inplace=True)
valid.fillna(0, inplace=True)

print train.shape, valid.shape


def XGBoost_(train=train, y=trainy, test=None, valid=valid, validy=validy, k=0, num_round=500,
             gamma=0.02, min_child_weight=1.1, max_depth=8, lamda=20, scale_pos_weight=13,
             subsamp=0.7, col_bytree=0.7, col_bylevel=0.7, eta=0.01, file="aac"):
    param = {'booster': 'gbtree',
             'objective': 'reg:logistic',
             # 'eval_metric':'auc',
             # 'scale_pos_weight':scale_pos_weight,
             # 'gamma': gamma,
             'min_child_weight': min_child_weight,
             'max_depth': int(max_depth),
             'lambda': int(lamda),
             'subsample': subsamp,
             'colsample_bytree': col_bytree,
             'colsample_bylevel': col_bylevel,
             # 'eta': 0.08,
             'eval_metric': 'logloss',
             # 'tree_method': 'exact',
             'seed': 0,
             'nthread': 8}
    dtrain = xgb.DMatrix(train, label=y)
    dvalid = xgb.DMatrix(valid, label=validy)
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    # auc = cv_log['test-auc-mean'].max()
    bst = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=8)
    # make prediction


    scores = bst.predict(dvalid, ntree_limit=bst.best_ntree_limit)
    ps = bst.predict(dtrain, ntree_limit=bst.best_ntree_limit)
    lgloss = logloss(validy, scores)
    trlgloss = logloss(trainy, ps)

    print "logloss", lgloss
    bst.save_model('./bayes/bst{}.model'.format(int(lgloss * 10000)))

    fp, tp, thresholds = metrics.roc_curve(validy, scores, pos_label=1)
    auc = metrics.auc(fp, tp)
    print "AUC:{}".format(auc)

    # get feature score
    feature_score = bst.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
    fs = []
    for (key, value) in feature_score:
        fs.append("{0},{1}\n".format(key, value))
    ff = './bayes/afeaturesImportance{0}.csv'.format(lgloss)
    with open(ff, 'w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)

    with open('./axgb.txt', mode='a+') as f:
        f.write('trlogloss:{0} logloss:{1} AUC:{2} ntree:{3}\n'.format(trlgloss, lgloss, auc, bst.best_ntree_limit))
        f.write(str(param))
        f.write("\n")

    return -lgloss


from bayes_opt import bayesian_optimization

xgbBo = bayesian_optimization.BayesianOptimization(XGBoost_, {
    # "gamma":(0.01,1),
    'min_child_weight': (.0, 500),
    'max_depth': (5, 15),
    'lamda': (5, 500),
    'subsamp': (0.5, 0.9),
    'col_bytree': (0.5, 0.9),
    'col_bylevel': (0.5, 0.9),
    # 'eta':(0.01,1)
})

xgbBo.maximize()

del train, valid
