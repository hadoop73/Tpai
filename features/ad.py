# coding:utf-8

from pyspark import SparkContext,SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.catalog import Column
from pyspark.sql.types import *
from pyspark.sql import functions
from pyspark.sql.functions import udf
from datetime import datetime,timedelta
import os
from pyspark.sql.window import Window
from pyspark.storagelevel import StorageLevel
from pyspark.sql.functions import UserDefinedFunction,rank, col
from pyspark.sql.types import LongType,StringType
import numpy as np
import pandas as pd


os.environ["SPARK_HOME"] = "/home/hadoop/spark-2.0.1-bin-hadoop2.7"   #KeyError: 'SPARK_HOME'
conf = SparkConf()
conf.set("spark.hadoop.validateOutputSpecs", "false")
conf.setMaster('spark://master:7077')
sc=SparkContext(appName='Tpai',conf=conf)
sc.setLogLevel('warn')
sqlContext = SQLContext(sc)

ad = sqlContext.read.format('com.databricks.spark.csv') \
    .options(header='true', charset="utf-8") \
    .load('hdfs://192.168.1.118:9000/home/hadoop/dup/ad.csv')

train = sqlContext.read.format('com.databricks.spark.csv') \
    .options(header='true', charset="utf-8") \
    .load('hdfs://192.168.1.118:9000/home/hadoop/dup/train.csv')

train26 = sqlContext.read.format('com.databricks.spark.csv') \
    .options(header='true', charset="utf-8") \
    .load('hdfs://192.168.1.118:9000/home/hadoop/dup/train26.csv')



def adf(dd=None,col='',prefix=''):
    data1 = dd.filter("label=1").groupBy(col).count()
    data1 = data1.withColumnRenamed('count',col+'count1')
    data0 = dd.filter("label=0").groupBy(col).count()
    data0 = data0.withColumnRenamed('count',col+'count0')

    data = data0.join(data1,on=col,how='outer')
    data = data.fillna(0)

    print data.show()
    data.toPandas().to_csv('../data/ad/train_ad_{0}_{1}.csv'.format(col,prefix),index=None)

d = train.join(ad,on='creativeID',how='left')
d26 = train26.join(ad,on='creativeID',how='left')

for col in ['adID','camgaignID','appID','appPlatform','advertiserID']:
    adf(d,col,prefix='all')
    adf(d26,col,prefix='26')
del d,d26


user = sqlContext.read.format('com.databricks.spark.csv') \
    .options(header='true', charset="utf-8") \
    .load('hdfs://192.168.1.118:9000/home/hadoop/dup/user.csv')



def userDummy(dd=None,col='',prefix=''):
    data1 = dd.filter("label=1").groupBy(col).count()
    data1 = data1.withColumnRenamed('count',col+'count1')
    data0 = dd.filter("label=0").groupBy(col).count()
    data0 = data0.withColumnRenamed('count',col+'count0')

    data = data0.join(data1,on=col,how='outer')
    data = data.fillna(0)

    print data.show()

    data.toPandas().to_csv('../data/user/train_user_{}_{}.csv'.format(col,prefix),index=None)

d = train.join(user,on='userID',how='left')
d26 = train26.join(user,on='userID',how='left')

for col in ['gender','education','marriageStatus',
            'haveBaby']:
    userDummy(d,col,prefix='all')
    userDummy(d26,col,prefix='26')









