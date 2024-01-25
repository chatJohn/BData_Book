# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession
from pyspark import SparkContext
try:
    sc.stop()
except:
    pass
from pyspark.sql.types import StructType, StructField, StringType, Row
from pyspark import Row
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

#统计各大二手车品牌的数量
def to_pair(item):
    items = item.split(",")
    car_size = items[6]
    try:
        sale_count = int(items[10])
    except Exception as e:
        sale_count = 0
    return car_size, sale_count


if __name__ == "__main__":
    sc = SparkContext(master='yarn',appName='CarSizeRDD')
    spark = SparkSession.builder.getOrCreate()
    data_rdd = sc.textFile("car_info - all_data - washed.csv")
    filtered_rdd = data_rdd.zipWithIndex().filter(lambda x: x[1] > 0).map(lambda x: x[0])
    pair_rdd = filtered_rdd.map(lambda x: to_pair(x)).reduceByKey(
        lambda x, y: x + y).map(lambda x: Row(CarSize=x[0], Sales=x[1])).sortBy(
        lambda x: x[1], False)


    schema = StructType([StructField("CarSize", StringType(), True), StructField("Sales", StringType(), True)])
    groupCarSize_df=spark.createDataFrame(pair_rdd)
    groupCarSize_df.show()

    # 使用Pandas绘制饼状图
    df_pandas = groupCarSize_df.toPandas().head(10)
    font_path = 'wqy-zenhei.ttc'  # 替换为你的微软雅黑字体文件路径
    font_properties = FontProperties(fname=font_path)
    df_pandas.plot(kind='pie', y='Sales', labels=df_pandas['CarSize'], autopct='%1.1f%%',
                   textprops={'fontproperties': font_properties})

    plt.title('二手车车型分布\n\n', fontproperties=font_properties)
    plt.axis('equal')  # 使饼状图成为一个圆
    plt.legend().set_visible(False)
    plt.tight_layout()
    plt.show()



