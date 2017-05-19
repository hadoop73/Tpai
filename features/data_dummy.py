# coding:utf-8


import pandas as pd


data = pd.read_csv('../data/dup/train21all_ad_user.csv')

for col in ['gender','education','marriageStatus','connectionType','telecomsOperator',
            'haveBaby','appID','appPlatform','advertiserID']:
    t = pd.get_dummies(data[col],prefix=col)
    data = data.join(t)

cols = ['userID','gender','education','marriageStatus','conversionTime','positionID','hometown',
        'haveBaby','creativeID','adID','camgaignID','appID','appPlatform','advertiserID','residence',
        'connectionType','telecomsOperator']

data.drop(cols,axis=1,inplace=True)
#data.to_csv('../data/dup/train21all_ad_user_dummy.csv',index=None)

# 采样负样本，采样 26 27 日负样本
train0 = data[data['label']==0]
d = []
for day in [26,27]:
    t = train0[(train0['clickTime']>=day*10000)&(train0['clickTime']<day*10000+10000)]
    d.append(t)
# 采样正样本,小于 29 日的所有正样本
t = data[(data['label']==1)&(data['clickTime']<29*10000)]
d = pd.concat(d+[t])

d.drop('clickTime',axis=1,inplace=True)
d.fillna(-10,inplace=True)
d.to_csv('../data/dup/train_fea{}.csv'.format(21),index=None)
print d.head()
print "train{} size:".format(21),d.shape



# 采样验证数据集 valid21.csv
valid = data[(data['clickTime']>=29*10000)&(data['clickTime']<29*10000+10000)]
valid.reset_index(inplace=True,drop=True)
vd = pd.read_csv('../data/dup/valid21.csv')
print vd.head()
print "vd!!!!"
valid.loc[:,'label'] = vd.loc[:,'label']
valid.drop('clickTime',axis=1,inplace=True)
valid.fillna(-10,inplace=True)

valid.to_csv('../data/dup/valid_fea{}.csv'.format(21),index=None)
print valid.head()
print "valid{} size:".format(21), valid.shape


