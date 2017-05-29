# coding:utf-8


import subprocess, sys, os, time

start = time.time()

NR_THREAD = 6

# 首先对数据集进行采样，取 26 27 日作为训练数据，29作为验证数据集，21为生成的样本后缀。train21.csv valid21.csv
cmd = './features/data_sample.py 26 27 29 21'
#subprocess.call(cmd, shell=True)

# 采样 28 29 作为训练数据，预测 test 作为预测数据
cmd = './data_sample.py 28 29 30 22'
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
cmd ='./ad-a.py ../data/dup/dtest.csv ../data/dup/dtest_sparse.csv'
#subprocess.call(cmd, shell=True)

# 获得没有 header 的数据，输出为 dtrain1_.csv,dvalid1_.csv,dtest_.csv
cmd ='../features/train_data.py dtrain1 dvalid2 dtest'
#subprocess.call(cmd, shell=True)


cmd = './gbdt -t 13 -s 6 ./data/dup/valid.dense ./data/dup/valid.sparse ' \
      './data/dup/train.dense ./data/dup/train.sparse ' \
      './data/dup/valid.gbdt.out ./data/dup/tr.gbdt.out'
#subprocess.call(cmd, shell=True)

cmd = './gbdt -t 8 -s 6 ../data/dup/dtest_.csv ../data/dup/dtest_sparse.csv ' \
      '../data/dup/dtrain1_.csv ../data/dup/dtrain_sparse.csv ' \
      '../data/dup/te.gbdt.out ../data/dup/tra.gbdt.out'
#subprocess.call(cmd, shell=True)

cmd ='./ad-b.py ../data/dup/dtest1.csv ../data/dup/te.gbdt.out ../data/dup/te.ffm'
#subprocess.call(cmd, shell=True)

# 对测试数据 ffm 编码 va
cmd ='./ad-b.py ../data/dup/dvalid2.csv ../data/dup/va2.gbdt.out ../data/dup/va2.ffm'
#subprocess.call(cmd, shell=True)

# 对训练数据 ffm 编码，csv_path 需要有 header 的数据 train.ffm
cmd ='./ad-b.py ../data/dup/dtrain1.csv ../data/dup/tr.gbdt.out ../data/dup/train1.ffm'
#subprocess.call(cmd, shell=True)

cmd = './ffm-train -k 6 -t 20 -r 0.1 --auto-stop -s {nr_thread} -p ./data/dup/valid_part.ffm ./data/dup/train_part.ffm modelA'.format(nr_thread=NR_THREAD)
#subprocess.call(cmd, shell=True)

cmd = './ffm-train -k 4 -t 200 -s {nr_thread} -p ./data/dup/valid.ffm ./data/dup/train.ffm modelA'.format(nr_thread=NR_THREAD)
#subprocess.call(cmd, shell=True)

cmd = "./libFM -task c -train ./data/dup/train.rand.fm -test ./data/dup/test.rand.fm -dim '1,1,8' -iter 50 " \
      "-out ./te.out -method sgda -learn_rate 0.01 -init_stdev 0.1 -validation ./data/dup/valid.rand.fm"
#subprocess.call(cmd, shell=True)

cmd = "./libFM -task c -train ./data/dup/train.tmp.dense.fm -test ./data/dup/valid.tmp.dense.fm -dim '1,1,8' -iter 20 -out ./test.out -method mcmc -learn_rate 0.01 -init_stdev 0.1"
subprocess.call(cmd, shell=True)

"""
使用测试数据 te.ffm 通过模型 model 进行预测
"""
cmd = './ffm-predict ./data/dup/valid.ffm modelA ./data/dup/valid.out'.format(nr_thread=NR_THREAD)
#subprocess.call(cmd, shell=True)

cmd = './ffm-predict ./data/dup/test.ffm modelA ./data/dup/res.out'.format(nr_thread=NR_THREAD)
#subprocess.call(cmd, shell=True)

print 'time used = {0:.0f}'.format(time.time()-start)
