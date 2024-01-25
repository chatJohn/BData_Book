from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
# 创建了SparkSession实例,使用yarn作为集群管理器
spark = SparkSession.builder.master("yarn").appName("PredictionModel").getOrCreate()
# 读取数据集
data = spark.read.csv("vehicles_data_20231012.csv", header=True, inferSchema=True, sep=" ")
# 删除标签列，得到特征列
feature_cols = data.drop("车型报价").columns
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
trainData, testData = data.randomSplit([0.8, 0.2], seed=42)
trainData = assembler.transform(trainData)
testData = assembler.transform(testData)
# 创建回归模型
regressors = [
    RandomForestRegressor(labelCol="车型报价", featuresCol="features", maxDepth=20, maxBins=128),
    DecisionTreeRegressor(labelCol="车型报价", featuresCol="features", maxDepth=20, maxBins=128)
]
# 训练模型并预测
for model in regressors:
    regressor_model = model.fit(trainData)
    predictions = regressor_model.transform(testData)
    evaluator = RegressionEvaluator(labelCol="车型报价", predictionCol="prediction", metricName="mae")
    mean_absolute_error = evaluator.evaluate(predictions)
    evaluator = RegressionEvaluator(labelCol="车型报价", predictionCol="prediction", metricName="mse")
    mean_squared_error = evaluator.evaluate(predictions)
    # 输出各个模型的评估指标结果
    print(model)
    print("\tMean Absolute Error:", mean_absolute_error)
    print("\tMean Squared Error:", mean_squared_error)
    predictions.select("车型报价", "prediction").show(10)
spark.stop()

