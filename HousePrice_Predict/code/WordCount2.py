from pyspark import SparkConf,SparkContext
import re
#export HADOOP_ROOT_LOGGER=DEBUG,console
#conf = SparkConf().setMaster("local").setAppName("WordCount")
sc = SparkContext(conf = SparkConf()) #创建SparkContext对象
#text = sc.textFile("file:///home/cjh-lqx/projects/the_adventures_of_sherlock_holmes.txt")
text = sc.textFile("input/the_adventures_of_sherlock_holmes.txt") #加载文本数据
repl_pattern = re.compile(r'[^a-zA-Z]') 
words_init = text.flatMap(lambda line:repl_pattern.split(line)) #以非字母类型字符为分割符,将单词提取出来

words = words_init.filter(lambda line:len(line.strip())>0) #剔除空字符串
words_dict = words.map(lambda word : (word,1)) #将单词映射为键值对形式,以便后面进行统计
words_group = words_dict.groupByKey() #按键值分组

words_count = words_group.map(lambda count_list:(count_list[0],sum(count_list[1]))) #统计各个单词出现的次数

#words_count2 = words_dict.reduceByKey(lambda a,b:a+b) #以上两步可用这一步进行替代

total_words = sum(words_count.values().collect()) #统计总的不同的单词的个数

index = 0
def getIndex(): #构建各个不同的单词出现的序号
    global index
    index += 1
    return index
    
#为每个单词添加序号并将词频的统计结果存入结果目录中
count_result = words_count.map(lambda value : (getIndex(),value)) 
#count_result.saveAsTextFile("file:///home/cjh-lqx/projects/result")
count_result.saveAsTextFile("output/wordcount3")
