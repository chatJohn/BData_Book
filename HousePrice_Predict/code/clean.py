# -*- coding: utf-8 -*-

import requests,time,re,csv,os,pandas_profiling
from bs4 import BeautifulSoup as BS
from multiprocessing import Pool
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from urllib.request import quote
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PowerTransformer,PolynomialFeatures
from sklearn.linear_model import LinearRegression,LassoCV,LogisticRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import KFold,train_test_split,StratifiedKFold,GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,accuracy_score, \
                            precision_score,recall_score, roc_auc_score
import matplotlib

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

import warnings
warnings.filterwarnings('ignore')
#myfont = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\华文细黑 常规')

#print(matplotlib.matplotlib_fname())
# 读取数据，查看数据基本信息，查看哪些字段含有空值
df0 = pd.read_csv('data.csv')
print(df0.info()) # 由于数据尚未处理和转化，不使用describe()
df0.T.head(35)

# 去除无用或无数据信息,将所有的“暂无数据”全部转为空值
df1 = df0.copy().drop(['Unnamed: 0','房屋朝向','所在楼层','供暖方式','链家编号'], axis=1)
df1 = df1.replace('暂无数据', np.nan)

# 部分无名称楼盘基本为车位，暂时除去。可以在爬取过程中的“异常提示”检查具体网址。
df1 = df1[df1.楼盘名称.notnull()]

# 成交单价单位统一为万
df1.成交单价 = df1.成交单价/ 10000

# 字段取数值部分
df1.区域 = df1.区域.str.split('二手房', expand=True)[0]
df1.子区域 = df1.子区域.str.split('二手房', expand=True)[0]
df1.建筑面积 = df1.建筑面积.str.replace('㎡', '')
df1.套内面积 = df1.套内面积.str.replace('㎡', '')
'''
# 分离所在楼层和楼层数
df1.所在楼层 = df0.所在楼层.str.split("\(共",expand=True)[0]
df1 = df1.join(df0.所在楼层.str.split("\(共",expand=True)[1].str.replace('层\)',''))
df1.rename(columns={1:'楼层数'},inplace=True)
'''
# 转化时间序列
df1['挂牌时间'] = pd.to_datetime(df1['挂牌时间'])
df1['成交时间'] = pd.to_datetime(df1['成交时间'])
pd.to_datetime(df1['挂牌时间'])
pd.to_datetime(df1['成交时间'])

# 将合适的对象数据转为数值
df1[['经度', '纬度', '成交价', '成交单价', '挂牌价格', '成交周期','调价',
     '带看', '关注', '浏览', '建筑面积', '套内面积', '建成年代']]\
    = df1[['经度', '纬度', '成交价', '成交单价', '挂牌价格', '成交周期','调价',
           '带看', '关注', '浏览', '建筑面积', '套内面积', '建成年代']].apply(pd.to_numeric, errors='corece')

# 重设索引，检查清洗后数据基本信息，是否处理成功，是否含空值等
df1.reset_index(drop=True, inplace=True)
print(df1.info())
print(df1.describe().T)

# 生成一份已清洗csv数据文件
df1.to_csv('clean-data.csv')

print(df1[df1.建成年代>=2020])

#求异常楼盘同名称楼盘建成年代众数，以修复异常值。
df1.loc[3034,'建成年代'] = df1[df1.楼盘名称=='北城国际中心'].建成年代.mode()[0]


#检查是否修复成功
print('\n',df1.loc[3034,'建成年代'])

# 数据整体房源分布、成交单价初探
plt.figure(figsize=(8,6),dpi=300)
plt.grid(linestyle='--')
plt.title("成交房源经纬度、房价关系图",size=8)
plt.xlabel("经度",size=8)
plt.ylabel("纬度",size=8)
plt.tick_params(labelsize=7)
cc = df1.成交单价
clist = [i for i in cc]
plt.scatter(df1.经度,df1.纬度,c=clist,cmap='Reds',s=50,alpha=0.5)

# 查看整体数据中，挂牌和成交时间分布直方图
# 可以发现2015上半年及之前基本没有挂牌信息，爬取的数据主要集中在2018及2019年。
# 我们会简单对比2019和2018上半年情况，并着重分析2019h1的房源情况。
plt.figure(figsize=(8,6),dpi=300)
plt.grid(linestyle='--')
plt.title("挂牌和成交时间分布",size=8)
plt.ylabel("成交数",size=8)
plt.xlabel("时间",size=8)
df1.成交时间.hist(bins=40,label='成交时间')
df1.挂牌时间.hist(bins=40,label='挂牌时间')
plt.legend(fontsize=10)
plt.tick_params(labelsize=7)

df2019h1 = df1[(df1.成交时间>='20190101')&(df1.成交时间<='20190630')]    # 2019h1成交的房源信息
df2018h1 = df1[(df1.成交时间>='20180101')&(df1.成交时间<='20180630')]    # 2018h1成交的房源信息

df2019h1.to_csv('df2019h1.csv',index=False)
df2018h1.to_csv('df2018h1.csv',index=False)
plt.show()

# 用pyecharts作2019h1重庆二手房成交量日历图，周日周一比较火热。
import datetime,random
from pyecharts import options as opts
from pyecharts.charts import Calendar

begin = datetime.date(2019,1,1)
end = datetime.date(2019,6,30)
data = [[str(t)[:10],v] for t,v in df2019h1.成交时间.value_counts().sort_index().items()]

c = (
        Calendar(opts.InitOpts(width = '800px',height = '400px')) # width、height设置画布大小
        .add("", data, calendar_opts=opts.CalendarOpts(range_=['2019-01-01', '2019-06-30']))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Calendar-2019h1重庆二手房成交量"),
            visualmap_opts=opts.VisualMapOpts(
                max_=161,
                min_=0,
                orient="horizontal",
                is_piecewise=False,
                pos_top="220px",
                pos_left="50px",
        ),
    )
)
#生成html图表文件
#c.render('重庆二手房成交量.html')
# 在notebook中显示图表
#c.render_notebook()

# 成交单价方面渝中区首屈一指，而江北区已超过了中心区域的沙坪坝区。
df2019h1.boxplot(column='成交单价', by='区域',figsize=(16,9))
plt.grid(linestyle='--')
plt.title('各区成交单价箱型图',size=20)
plt.ylabel('万元')
plt.tick_params(labelsize=13)

plt.figure(figsize=(16,15),dpi=300)
plt.subplots_adjust(wspace =0.3, hspace =0.3)
plt.grid(linestyle='--')

plt.subplot(3,1,1)
plt.title('2019h1、2018h1成交周期kde图',size=15)
plt.xlabel('成交周期',size=11)
sns.kdeplot(df2019h1.成交周期,shade=True,label='2019h1')
sns.kdeplot(df2018h1.成交周期,shade=True,label='2018h1')

plt.subplot(3,1,2)
plt.title('2019h1、2018h1建筑面积kde图',size=15)
plt.xlabel('建筑面积',size=11)
plt.xticks(range(0,400,10))
sns.kdeplot(df2019h1.建筑面积,shade=True,label='2019h1')
sns.kdeplot(df2018h1.建筑面积,shade=True,label='2018h1')

plt.subplot(3,1,3)
plt.title('2019h1较2018h1各数值平均增量比例图',size=15)
compare = (df2019h1.mean()[2:]-df2018h1.mean()[2:])/df2018h1.mean()[2:]
compare.plot(kind='bar')
plt.tick_params(labelsize=12)
plt.xticks(rotation=0)
for a,b in zip(range(12),compare.values):
    plt.text(a,b,'%.3f'%b,ha = 'center',va = 'bottom',fontsize=12)



dfall = df2019h1.楼盘名称.count()
dfup = df2019h1.楼盘名称[df2019h1.成交价>df2019h1.挂牌价格].count()
dfdown = df2019h1.楼盘名称[df2019h1.成交价<df2019h1.挂牌价格].count()
dfsame = dfall-dfup-dfdown

plt.figure(dpi=300)
ratios=[dfup/dfall,dfdown/dfall,dfsame/dfall]#存放比例列表
colors=['gold','coral','orange'] # 存放颜色列表，与比例相匹配
labels=["加价",'减价','不变'] # 存放各类元素标签
explode=(0,0,0) # 炸开的比例
plt.pie(ratios,radius=1,explode=explode,colors=colors,
        labels=labels,autopct='%.2f%%',textprops = {'fontsize':10, 'color':'black'}) # 绘制饼图
plt.title('2019上半年成交二手房加减价比例',size=15)
plt.axis('equal') # 将饼图显示为正圆形

# 2018上半年成交二手房加减价比例：
# 我们再参考一下2018h1的情况，发现18年上半年减价成交占比更少，加价成交占比更多
dfall1 = df2018h1.楼盘名称.count()
dfup1 = df2018h1.楼盘名称[df2018h1.成交价>df2018h1.挂牌价格].count()
dfdown1 = df2018h1.楼盘名称[df2018h1.成交价<df2018h1.挂牌价格].count()
dfsame1 = dfall1-dfup1-dfdown1

plt.figure(dpi=300)
ratios=[dfup1/dfall1,dfdown1/dfall1,dfsame1/dfall1]#存放比例列表
colors=['gold','coral','orange'] # 存放颜色列表，与比例相匹配
labels=["加价",'减价','不变'] # 存放各类元素标签
explode=(0,0,0) # 炸开的比例
plt.pie(ratios,radius=1,explode=explode,colors=colors,
        labels=labels,autopct='%.2f%%',textprops = {'fontsize':10, 'color':'black'}) # 绘制饼图
plt.title('2018上半年成交二手房加减价比例',size=15)
plt.axis('equal') # 将饼图显示为正圆形

dfdelta = (df2019h1.成交价-df2019h1.挂牌价格)
# 2018h1平均加价，平均减价，挂牌后平均加减价
dfdelta1 = (df2018h1.成交价-df2018h1.挂牌价格)
print(dfdelta.mean(),dfdelta1.mean())
plt.figure(figsize=(16,10),dpi=180)
#plt.grid(linestyle='--')
plt.xlim(-1,2)
plt.title('2018h1/2019h1 成交加减价情况',size=20)
plt.ylabel('万元')
plt.tick_params(labelsize=13)
plt.bar(['2019h1','2018h1'],[dfdelta.mean(),dfdelta1.mean()],width=0.2,color='lightcoral')
for a,b in zip(range(2),[dfdelta.mean(),dfdelta1.mean()]):
    plt.text(a,b,'%.2f'%b,ha = 'center',va = 'bottom',fontsize=13)


plt.figure(figsize=(16,10),dpi=180)
plt.title('各区成交平均降价幅度',size=20)
plt.ylabel('万元')
de = df2019h1.成交价-df2019h1.挂牌价格
de.name = 'delta'
df2019h1.join(de).groupby('区域').delta.mean().sort_values().plot(kind='bar',color='cadetblue')
plt.tick_params(labelsize=13)
plt.xticks(rotation=0)

plt.figure(figsize=(16, 9),dpi=150)
corr = df2019h1.corr()
sns.heatmap(corr,cmap='Reds',annot=True)

# (3) 户型、装修成交情况：
# 成交数前5的户型中，前三名是"2室1厅1厨1卫","1室1厅1厨1卫","2室2厅1厨1卫"。
# 看来还是主流户型（性价比高）比较好卖呀！
plt.figure(figsize=(16,10),dpi=150)
plt.subplot(1,2,1)
plt.title('成交量前5户型柱状图',size=20)
plt.ylabel('成交数',size=11)
df2019h1.房屋户型.value_counts()[:5].plot(kind='bar',color='seagreen') # 成交数前5的户型
plt.tick_params(labelsize=8)
plt.xticks(rotation=0)

# 精装房最受欢迎。毛坯成交量最少，查阅数据可知其平均单价倒数第二，但平均总价却最高。
plt.subplot(1,2,2)
plt.title('装修情况',size=20)
plt.ylabel('成交数',size=11)
df2019h1.装修情况.value_counts().plot(kind='bar',color='tan')
print(df2019h1.groupby('装修情况').成交价.mean().sort_values(ascending=False))
print(df2019h1.groupby('装修情况').成交单价.mean().sort_values(ascending=False))
plt.tick_params(labelsize=13)
plt.xticks(rotation=0)

plt.show()

# WordCloud-楼盘热度:
# 哪些是热门楼盘？
from pyecharts import options as opts
from pyecharts.charts import Page, WordCloud
from pyecharts.globals import SymbolType
words = [(v,n) for v,n in df2019h1.楼盘名称.value_counts().items()]


w = (
        WordCloud()
        # word_size_range设置非常重要，最小值非负的话，低数值词组会填满形状外区域！
        .add("", words,shape='diamond',word_size_range=[-10, 40],word_gap=20)
        .set_global_opts(title_opts=opts.TitleOpts(title="楼盘热度"))
)
# word_size_range=[20, 100],rotate_step=90,shape='diamond'  opts.InitOpts(width = '1000px',height = '1000px')
w.render('楼盘热度.html')
w.render_notebook()