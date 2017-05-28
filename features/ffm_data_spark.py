#!/home/hadoop/env2.7/bin/python
# coding:utf-8

from pyspark import SparkContext,SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import DoubleType,FloatType,IntegerType,StringType
import os,hashlib,math


def hashstr(str, nr_bins=1e+6):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16)%(nr_bins-1)+1


freqfeas = {}
def freqFea(data,cols):
    for col in cols:
        d = data[col]
        d['count'] = 1
        d = d.groupby(col,as_index=False)['count'].sum()
        d = d[d['count']>=10]
        freqfeas[col] = set(d[col])

def logFun(x):
    #x = int(1000*x)
    x = int(x)
    if x<2:
        return "sp"+str(x)
    else:
        return str(int(math.log(float(x))**2))

os.environ["SPARK_HOME"] = "/home/hadoop/spark-2.0.1-bin-hadoop2.7"   #KeyError: 'SPARK_HOME'
conf = SparkConf()
conf.set("spark.hadoop.validateOutputSpecs", "false")
conf.setMaster('spark://master:7077')
sc=SparkContext(appName='Tpai',conf=conf)
sc.setLogLevel('warn')
sqlContext = SQLContext(sc)


train = sqlContext.read.format('com.databricks.spark.csv') \
    .options(header='true', charset="utf-8") \
    .load('hdfs://192.168.1.118:9000/home/hadoop/dup/train_xgb113U.csv')

df = sqlContext.read.format('com.databricks.spark.csv') \
    .options(header='true', charset="utf-8") \
    .load('hdfs://192.168.1.118:9000/home/hadoop/dup/train_xgb113U_df.csv')

y = train[['label']]
train = train.drop('label')

cols = ['adID', 'camgaignID', 'appID', 'appPlatform', 'advertiserID', 'creativeID', 'sitesetID',
        'positionType', 'positionID', 'telecomsOperator', 'connectionType', 'gender',
        'education', 'marriagedStatus', 'haveBaby', 'hometown', 'residence', 'liveState']
d = train
cols = [col for col in cols if col in d.columns]
freqFea(d)
n = 0
for col in d.columns:
    if col in cols:
        d.loc[:, col] = d[col].apply(
            lambda x: "{0}:{1}:1".format(n, hashstr(str(x)) if x in freqfeas[col] else hashstr("sparse" + str(x))))
    elif d[col].max() < 1:
        d.loc[:, col] = d[col].apply(lambda x: "{0}:{1}:1".format(n, logFun(1000 * x)))
    else:
        d.loc[:, col] = d[col].apply(lambda x: "{0}:{1}:1".format(n, logFun(x)))

for col in df.columns:
    df.loc[:, col] = df[col].apply(
        lambda x: "{0}:{1}:1".format(n, hashstr(str(x))))

df = d.join(df)
d = y.join(df)
print d.head()
print d.shape
d.toPandas().to_csv('../data/dup/train.ffm')
