from pyspark.sql import SQLContext
from pyspark.sql import Row
from pyspark import SparkContext
from pyspark.ml.clustering import KMeans,KMeansModel
from pyspark.ml.linalg import Vectors
#将列表转化为20维向量
def f(x):
    rel = {}
    rel['features'] = Vectors.dense(float(x[0]),float(x[1]),float(x[2]),float(x[3]),float(x[4]),
    	float(x[5]),float(x[6]),float(x[7]),float(x[8]),float(x[9]),float(x[10]),float(x[11]),
    	float(x[12]),float(x[13]),float(x[14]),float(x[15]),float(x[16]),float(x[17]),float(x[18]),
    	float(x[19]))
    return rel
#建立sparkcontext对象
sc = SparkContext('local','test')
sqlContext = SQLContext(sc)
df = sc.textFile("data.csv").map(lambda line: line.split(',')).map(lambda p: Row(**f(p))).toDF()
#调用库函数进行kmeans聚类
kmeansmodel = KMeans().setK(8).setFeaturesCol('features').setPredictionCol('prediction').fit(df)
results = kmeansmodel.transform(df).collect()
for item in results:
     print(str(item[0])+' is predcted as cluster'+ str(item[1]))
