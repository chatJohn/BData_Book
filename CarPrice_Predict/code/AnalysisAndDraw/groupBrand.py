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
from wordcloud import WordCloud

#统计各大二手车品牌的数量
def to_pair(item):
    items = item.split(",")
    brand_name = items[5]
    try:
        sale_count = int(items[10])
    except Exception as e:
        sale_count = 0
    return brand_name, sale_count


if __name__ == "__main__":
    sc = SparkContext(master='local',appName='BrandRDD')
    spark = SparkSession.builder.getOrCreate()
    data_rdd = sc.textFile("Car_Data\car_info - all_data - washed.csv")
    filtered_rdd = data_rdd.zipWithIndex().filter(lambda x: x[1] > 0).map(lambda x: x[0])
    pair_rdd = filtered_rdd.map(lambda x: to_pair(x)).reduceByKey(
        lambda x, y: x + y).map(lambda x: Row(
        Brand=x[0], Sales=x[1])).sortBy(lambda x: x[1], False)


    schema = StructType([StructField("Brand", StringType(), True), StructField("Sales", StringType(), True)])
    groupBrand_df=spark.createDataFrame(pair_rdd)
    groupBrand_df.show(10)
    print(type(groupBrand_df))

    # 柱状图绘制
    df_pandas = groupBrand_df.toPandas().head(10)
    ax = df_pandas.plot(kind='bar', x='Brand', y='Sales')
    font_path = 'wqy-zenhei.ttc'  # 替换为你的微软雅黑字体文件路径
    font_properties = FontProperties(fname=font_path)
    plt.xticks(rotation=45)
    font_dict = {'fontsize': 12, 'fontweight': 'bold', 'color': 'black'}

    plt.xticks(range(len(df_pandas['Brand'])), df_pandas['Brand'], fontproperties=font_properties)
    for i, v in enumerate(df_pandas['Sales']):
        plt.text(i, v, str(v), ha='center', va='bottom')
    plt.title('二手车品牌前十在售量', fontproperties=font_properties)
    plt.xlabel('Brand')
    plt.ylabel('Sales')
    plt.tight_layout()
    plt.show()

    # 词云绘制
    df_pandas = groupBrand_df.toPandas()
    word_frequencies = {row['Brand']: row['Sales'] for _, row in df_pandas.iterrows()}
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          font_path='wqy-zenhei.ttc').generate_from_frequencies(word_frequencies)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title('Brand Word Cloud\n')
    plt.show()






