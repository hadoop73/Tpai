# coding:utf-8

import pandas as pd

import xgboost as xgb
#from xgboost import XGBClassifier as xgb
import numpy as np
from sklearn import metrics
import scipy as sp
import zipfile

def logloss(act, pred):
  epsilon = 1e-15
  pred = sp.maximum(epsilon, pred)
  pred = sp.minimum(1-epsilon, pred)
  ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
  ll = ll * -1.0/len(act)
  return ll

def evalerror(preds,dtrain):
    labels = dtrain.get_label()
    preds = preds/(preds+(1.0-preds)/0.1)
    return 'loss',logloss(np.array(labels),preds)

""""""
#atrain = pd.read_csv('../data/dup/train_xgbA.csv')
cols = ['userID','clickTime','home', 'hometown', 'residence', 'resid', 'positionID',
        'creativeID', 'adID', 'camgaignID', 'advertiserID', 'appID']

cols2 = ['education', 'haveBaby', 'positionType', 'connectionType']

cols3 = ['gender', 'marriageStatus', 'sitesetID', 'appPlatform', 'telecomsOperator']

cols += cols2 + cols3 + ['label']
atrain = pd.read_csv('../data/dup/train_3p_.csv')

cols = [c for c in atrain.columns if c in cols]


atrain.fillna(0,inplace=True)
trainy = atrain['label']
atrain.drop(cols,axis=1,inplace=True)

test = pd.read_csv('../data/dup/test_3p_.csv')
test.fillna(-10,inplace=True)
test.drop(cols,axis=1,inplace=True)

avalid = pd.read_csv('../data/dup/valid_3p_.csv')
avalid.fillna(-10,inplace=True)

validy = avalid['label']
avalid.drop(cols,axis=1,inplace=True)

#atest = pd.read_csv('../data/dup/test_xgb.csv')
#testy = atest['label']
#atest.drop('label',axis=1,inplace=True)




"""
    由于这里只是训练模型，并没有用到 test
   {'eval_metric': 'logloss', 'colsample_bylevel': 0.53004952590015819,
   'seed': 0, 'tree_method': 'exact', 'booster': 'gbtree',
   'colsample_bytree': 0.56764321396074324, 'nthread': 8, 'min_child_weight': 221.44444222886625,
   'objective': 'binary:logistic', 'max_depth': 9, 'lambda': 114}


"""
def XGBoost_(train=atrain,y=trainy,test=test,valid=avalid,validy=validy,k=0,num_round=300,
			 gamma=.658,min_child_weight=221.4,max_depth=9,lamda=114,scale_pos_weight=40,
			 subsamp=0.867,col_bytree=0.568,col_bylevel=0.53,eta=0.08,file="aac"):
    param = {'booster': 'gbtree',
             'objective': 'binary:logistic',
             # 'eval_metric':'auc',
             #'scale_pos_weight':int(scale_pos_weight),
             #'gamma': gamma,
             'min_child_weight': min_child_weight,
             'max_depth': int(max_depth),
             'lambda': int(lamda),
             #'subsample': subsamp,
             'colsample_bytree': col_bytree,
             'colsample_bylevel': col_bylevel,
             #'eta': eta,
             'eval_metric': 'logloss',
             'tree_method': 'exact',
             'seed': 0,
             'nthread': 8}
    dtrain = xgb.DMatrix(train, label=y,missing=-10)
    #del train
    dvalid = xgb.DMatrix(valid, label=validy,missing=-10)
    #del valid
    dtest = xgb.DMatrix(test, missing=-10)
    #del test
    watchlist = [(dtrain, 'train'),(dvalid, 'eval')]
    # auc = cv_log['test-auc-mean'].max()
    bst = xgb.train(param, dtrain, num_round, watchlist,early_stopping_rounds=10)
    # make prediction
    treeIndex = bst.predict(dvalid, ntree_limit=bst.best_ntree_limit,pred_leaf=True)
    df = pd.DataFrame(treeIndex)
    df.to_csv('../data/dup/valid.xgb.csv',index=None)

    treeIndex = bst.predict(dtrain, ntree_limit=bst.best_ntree_limit,pred_leaf=True)
    df = pd.DataFrame(treeIndex)
    df.to_csv('../data/dup/train.xgb.csv', index=None)
    #del train, valid
    #del dtrain,dvalid

    preds = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
    t = pd.read_csv('../data/dup/test.csv')
    t['prob'] = preds
    t[['instanceID', 'prob']].to_csv('./submission.csv', index=None)
    with zipfile.ZipFile('submission1023.zip', 'w') as fout:
        fout.write('submission.csv', compress_type=zipfile.ZIP_DEFLATED)

    treeIndex = bst.predict(dtest, ntree_limit=bst.best_ntree_limit, pred_leaf=True)
    df = pd.DataFrame(treeIndex)
    df.to_csv('../data/dup/test.xgb.csv', index=None)


    """
    t = pd.read_csv('../data/dup/test.csv')
    t['prob'] = preds
    t[['instanceID', 'prob']].to_csv('./submission.csv', index=None)

    lgloss = logloss(validy, scores)
    print "logloss", lgloss
    bst.save_model('bst{}.model'.format(str(lgloss)))
    with open('./record.xgbB','a+') as f:
        f.write('logloss: '+str(lgloss)+"\n")
        f.write(str(param)+"\n")

    #pro = pd.DataFrame({'prob': preds})
    #pro.to_csv('../data/res{}.csv'.format(str(lgloss)), index=None)

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
    return -lgloss"""


XGBoost_()


#predict()

