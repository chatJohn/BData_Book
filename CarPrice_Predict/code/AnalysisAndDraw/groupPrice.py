from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, Row
from pyspark import Row
from pyspark import SparkContext
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

#统计各个二手车的价格
def to_pair(item):
    items = item.split(",")
    try:
        price = float(items[1])
        sale_count = int(items[10])
    except Exception as e:
        price = 0
        sale_count = 0
    return price,sale_count

def dividePrice(m,n):
    if n==10:
        string = "10万以下"
    elif n==30:
        string = "10-30万"
    elif n==50:
        string = "30-50万"
    elif n==70:
        string = "50-70万"
    elif n==90:
        string = "70-90万"
    elif n==110:
        string = "90-110万"
    elif n==150:
        string = "110-150万"
    elif n==200:
        string = "150-200万"
    else:
        string = "200万以上"

    pair_rdd= (filtered_rdd.map(lambda x: to_pair(x)).filter(lambda x: m< x[0] < n).map(
        lambda x: (string, x[1])).sortBy(lambda x: x[1], False)
        .reduceByKey(lambda x, y: x + y).map(
        lambda x: Row(Price=x[0], Sales=x[1])))
    return pair_rdd

if __name__ == "__main__":
    sc = SparkContext(master='yarn', appName='PriceRDD')
    spark = SparkSession.builder.getOrCreate()
    data_rdd = sc.textFile("car_info - all_data - washed.csv")
    filtered_rdd = data_rdd.zipWithIndex().filter(lambda x: x[1] > 0).map(lambda x: x[0])
    for i in range(0,9):
        if i == 0:
            price_rdd = dividePrice(0,10)
        elif i > 0 and i < 6:
            pair_rdd = dividePrice(10*(2*i-1), 10*(2*i+1))
        elif i == 6:
            pair_rdd = dividePrice(110,150)
        elif i == 7:
            pair_rdd = dividePrice(150,200)
        elif i == 8:
            pair_rdd = dividePrice(200, 10000)

        if i!=0:
            price_rdd = price_rdd.union(pair_rdd)



    schema = StructType([StructField("Price", StringType(), True), StructField("Sales", StringType(), True)])
    groupPrice_df = spark.createDataFrame(price_rdd)
    groupPrice_df.show()

    # 画图
    df_pandas = groupPrice_df.toPandas()
    ax = df_pandas.plot(kind='bar', x='Price', y='Sales')
    font_path = 'wqy-zenhei.ttc'  # 替换为你的微软雅黑字体文件路径
    font_properties = FontProperties(fname=font_path)
    plt.xticks(rotation=45)
    font_dict = {'fontsize': 12, 'fontweight': 'bold', 'color': 'blue'}
    plt.xticks(range(len(df_pandas['Price'])), df_pandas['Price'], fontproperties=font_properties)
    plt.title('二手车价格分布',fontproperties=font_properties)
    plt.xlabel('Price')
    plt.ylabel('Sales')
    plt.tight_layout()
    plt.show()






