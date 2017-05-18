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


os.environ["SPARK_HOME"] = "/home/hadoop/spark-2.0.1-bin-hadoop2.7"   #KeyError: 'SPARK_HOME'
conf = SparkConf()
conf.set("spark.hadoop.validateOutputSpecs", "false")

conf.setMaster('spark://master:7077')
sc=SparkContext(appName='Tpai',conf=conf)
sqlContext = SQLContext(sc)


def mergeData():
    train = sqlContext.read.format('com.databricks.spark.csv')\
        .options(header='true', charset="utf-8")\
        .load('hdfs://192.168.1.118:9000/home/hadoop/dup/train.csv')
    ad = sqlContext.read.format('com.databricks.spark.csv')\
        .options(header='true',charset='utf-8')\
        .load('hdfs://192.168.1.118:9000/home/hadoop/dup/ad.csv')
    data = train.join(ad,on='creativeID')
    u = sqlContext.read.format('com.databricks.spark.csv') \
        .options(header='true', charset='utf-8') \
        .load('hdfs://192.168.1.118:9000/home/hadoop/dup/user.csv')
    data = data.join(u, on='userID')
    print data.show()
    data.write\
        .csv('hdfs://192.168.1.118:9000/home/hadoop/dup/train_ad_u.csv',
             header=True,mode='overwrite')


#mergeData()

d = sqlContext.read.format('com.databricks.spark.csv') \
    .options(header='true', charset="utf-8") \
    .load('hdfs://192.168.1.118:9000/home/hadoop/dup/train_ad.csv')

u = sqlContext.read.format('com.databricks.spark.csv') \
    .options(header='true', charset='utf-8') \
    .load('hdfs://192.168.1.118:9000/home/hadoop/dup/user.csv')

data = d.join(u,on='userID')
print data.show()
data.write\
        .csv('hdfs://192.168.1.118:9000/home/hadoop/dup/train_ad_u.csv',
             header=True,mode='overwrite')


