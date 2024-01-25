# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession
from pyspark import SparkContext
try:
    sc.stop()
except:
    pass
from pyspark.sql.types import StructType, StructField, StringType, Row
from pyspark import Row
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def to_pair(item):
    items = item.split(",")
    # print(items)
    power = items[7]
    try:
        sale_count = int(items[10])
    except Exception as e:
        sale_count = 0
    return power, sale_count


if __name__ == "__main__":
    sc = SparkContext(master='yarn',appName='PowerRDD')
    spark = SparkSession.builder.getOrCreate()


    data_rdd = sc.textFile("car_info - all_data - washed.csv")
    filtered_rdd = data_rdd.zipWithIndex().filter(lambda x: x[1] > 0).map(lambda x: x[0])
    pair_rdd = filtered_rdd.map(lambda x: to_pair(x)).reduceByKey(
        lambda x, y: x + y).map(lambda x: Row(Power=x[0], Sales=x[1])).sortBy(
        lambda x: x[1], False)


    schema = StructType([StructField("Power", StringType(), True), StructField("Sales", StringType(), True)])
    groupPower_df=spark.createDataFrame(pair_rdd)
    groupPower_df.show(10)

    # 画图
    df_pandas = groupPower_df.toPandas()
    ax = df_pandas.plot(kind='bar', x='Power', y='Sales' ,color='green')
    font_path = 'wqy-zenhei.ttc'  # 替换为你的微软雅黑字体文件路径
    font_properties = FontProperties(fname=font_path)
    plt.xticks(rotation=45)
    font_dict = {'fontsize': 12, 'fontweight': 'bold', 'color': 'black'}

    plt.xticks(range(len(df_pandas['Power'])), df_pandas['Power'], fontproperties=font_properties)
    plt.title('二手车能源类型分布图', fontproperties=font_properties)
    plt.xlabel('Power')
    plt.ylabel('Sales')
    plt.tight_layout()
    plt.show()



