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

def evalerror(preds,dtrain):
    labels = dtrain.get_label()
    preds = preds/(preds+(1.0-preds)/0.1)
    return 'loss',logloss(np.array(labels),preds)

""""""
#atrain = pd.read_csv('../data/dup/train_xgbA.csv')

atrain = pd.read_csv('../data/dup/train_xgb11U.csv')
atrain.fillna(-10,inplace=True)
trainy = atrain['label']
atrain.drop('label',axis=1,inplace=True)

test = pd.read_csv('../data/dup/test_xgb11U.csv')
test.fillna(0,inplace=True)
test.drop('label',axis=1,inplace=True)

avalid = pd.read_csv('../data/dup/valid_xgb11U.csv')
avalid.fillna(-10,inplace=True)

validy = avalid['label']
avalid.drop('label',axis=1,inplace=True)

#atest = pd.read_csv('../data/dup/test_xgb.csv')
#testy = atest['label']
#atest.drop('label',axis=1,inplace=True)




"""
    由于这里只是训练模型，并没有用到 test
    {'eval_metric': 'logloss', 'colsample_bylevel': 0.90000000000000002,
     'scale_pos_weight': 40, 'seed': 0, 'tree_method': 'exact', 'booster': 'gbtree',
     'colsample_bytree': 0.90000000000000002, 'nthread': 8, 'min_child_weight': 2.0,
     'subsample': 0.90000000000000002, 'eta': 0.010000000000000009, 'objective': 'binary:logistic',
      'max_depth': 10, 'gamma': 1.0, 'lambda': 15}

"""
def XGBoost_(train=atrain,y=trainy,test=test,valid=avalid,validy=validy,k=0,num_round=50,
			 gamma=1,min_child_weight=0.8,max_depth=8,lamda=15,scale_pos_weight=40,
			 subsamp=0.9,col_bytree=0.9,col_bylevel=0.62,eta=0.02,file="aac"):
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
    watchlist = [(dvalid, 'eval')]
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

    treeIndex = bst.predict(dtest, ntree_limit=bst.best_ntree_limit,pred_leaf=True)
    #p = bst.predict(dtrain, ntree_limit=bst.best_ntree_limit)
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


trainall = pd.read_csv('../data/dup/train_all_xgb.csv')
y = trainall['label']
trainall.drop('label',axis=1,inplace=True)

avalid = pd.read_csv('../data/dup/valid_xgb.csv')
vy = avalid['label']
avalid.drop('label',axis=1,inplace=True)

def predict(trainall=trainall,avalid=avalid):
    bst = xgb.Booster({'nthread': 8})
    bst.load_model('./bst117.model')
    dtrain = xgb.DMatrix(trainall)
    probdf = bst.predict(dtrain,ntree_limit=94,pred_leaf=True)

    dvalid = xgb.DMatrix(avalid)
    vdf = bst.predict(dvalid,ntree_limit=94, pred_leaf=True)
    del dtrain,dvalid
    df = pd.DataFrame(probdf)
    df.columns = ['pred'+str(col) for col in df.columns]
    trainall = trainall.join(df)

    vdf = pd.DataFrame(vdf)
    vdf.columns = ['pred' + str(col) for col in vdf.columns]
    avalid = avalid.join(vdf)
    df['label'] = y
    trainall['label'] = y
    avalid['label'] = vy
    print trainall.head()
    train = trainall[(trainall['clickTime'] >= 260000) & (trainall['clickTime'] < 280000)]
    del trainall
    import gc
    for col in df.columns:
        if col not in ['label']:
            d = df[['label',col]]
            print d.head()
            d = d.groupby(col,as_index=False)['label'].mean()
            d.rename(columns={'label':col+"mean_ratio"},inplace=True)
            train = train.merge(d,on=col,how='left')
            train.drop(col,axis=1,inplace=True)
            avalid = avalid.merge(d, on=col, how='left')
            avalid.drop(col,axis=1,inplace=True)
            del d
            gc.collect()
    print train.head()
    print train.shape

    train.to_csv('../data/dup/train_xgbLRA.csv', index=None)
    avalid.to_csv('../data/dup/valid_xgbLRA.csv', index=None)

#predict()

