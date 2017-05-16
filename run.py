# coding:utf-8


import subprocess, sys, os, time

start = time.time()

NR_THREAD = 6

# 首先对数据集进行采样
cmd = './data_sample.py 17 21 26 22 27 1'
#subprocess.call(cmd, shell=True)
# 构造新的验证集
cmd = './data_sample.py 17 21 26 24 29 2'
#subprocess.call(cmd, shell=True)

# 得到训练数据和验证数据的特征，输出为 dtrain1.csv dvalid1.csv dtest.csv
cmd = './get_features.py train1 valid2 test'
#subprocess.call(cmd, shell=True)

# 对训练数据集进行 稀疏特征和密集特征 抽取
cmd ='./ad-a.py ../data/dup/dtrain1.csv ../data/dup/dtrain_sparse.csv'
#subprocess.call(cmd, shell=True)

# 对验证数据集处理
cmd ='./ad-a.py ../data/dup/dvalid2.csv ../data/dup/dvalid2_sparse.csv'
#subprocess.call(cmd, shell=True)

# 获得测试数据的稀疏数据
cmd ='./ad-a.py ../data/dup/dtest1.csv ../data/dup/dtest_sparse.csv'
#subprocess.call(cmd, shell=True)

# 获得没有 header 的数据，输出为 dtrain1_.csv,dvalid1_.csv,dtest1_.csv
cmd ='../features/train_data.py dtrain1 dvalid2 dtest1'
#subprocess.call(cmd, shell=True)


cmd = './gbdt -t 18 -s 6 ../data/dup/dvalid2_.csv ../data/dup/dvalid2_sparse.csv ' \
      '../data/dup/dtrain1_.csv ../data/dup/dtrain_sparse.csv ' \
      '../data/dup/va2.gbdt.out ../data/dup/tr.gbdt.out'
#subprocess.call(cmd, shell=True)

cmd = './gbdt -t 8 -s 6 ../data/dup/dtest_.csv ../data/dup/dtest_sparse.csv ' \
      '../data/dup/dtrain1_.csv ../data/dup/dtrain_sparse.csv ' \
      '../data/dup/te.gbdt.out ../data/dup/tra.gbdt.out'
#subprocess.call(cmd, shell=True)

# 对训练数据 ffm 编码，csv_path 需要有 header 的数据 train.ffm
cmd ='./ad-b.py ../data/dup/dtrain1.csv ../data/dup/tr.gbdt.out ../data/dup/train1.ffm'
#subprocess.call(cmd, shell=True)

# 对测试数据 ffm 编码 va
cmd ='./ad-b.py ../data/dup/dvalid2.csv ../data/dup/va2.gbdt.out ../data/dup/va2.ffm'
#subprocess.call(cmd, shell=True)

cmd ='./ad-b.py ../data/dup/dtest1.csv ../data/dup/te.gbdt.out ../data/dup/te.ffm'
#subprocess.call(cmd, shell=True)

cmd = './ffm-train -k 4 -t 5 -l 0.0008 -s {nr_thread} -p ../data/dup/va2.ffm ../data/dup/train1.ffm model'.format(nr_thread=NR_THREAD)
subprocess.call(cmd, shell=True)

cmd = './ffm-train -k 4 -t 4 -s {nr_thread} -p ../data/dup/ta.ffm ../data/dup/train.ffm model'.format(nr_thread=NR_THREAD)
#subprocess.call(cmd, shell=True)

"""
使用测试数据 te.ffm 通过模型 model 进行预测
"""
cmd = './ffm-predict ../data/dup/va2.ffm model ../data/dup/va2.out'.format(nr_thread=NR_THREAD)
subprocess.call(cmd, shell=True)

cmd = './ffm-predict ../data/dup/te.ffm model ../data/dup/te.out'.format(nr_thread=NR_THREAD)
#subprocess.call(cmd, shell=True)

print 'time used = {0:.0f}'.format(time.time()-start)
