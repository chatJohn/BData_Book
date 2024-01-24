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

import warnings
warnings.filterwarnings('ignore')

df1 = pd.read_csv('clean-data.csv')
'''
# 数据整体房源分布、成交单价初探
plt.figure(figsize=(16,9),dpi=300)
plt.grid(linestyle='--')
plt.title('成交房源经纬度、房价关系图',size=20)
plt.xlabel('经度')
plt.ylabel('纬度')
cc = df1.成交单价
clist = [i for i in cc]
plt.scatter(df1.经度,df1.纬度,c=clist,cmap='Reds',s=50,alpha=0.5)
plt.show()
'''
'''地图
img = plt.imread('房价柱状图.png')
plt.figure(dpi=300)
plt.imshow(img)
'''

# 查看整体数据中，挂牌和成交时间分布直方图
# 可以发现2015上半年及之前基本没有挂牌信息，爬取的数据主要集中在2018及2019年。
# 我们会简单对比2019和2018上半年情况，并着重分析2019h1的房源情况。
plt.figure(figsize=(16,9),dpi=300)
plt.grid(linestyle='--')
plt.title('挂牌和成交时间分布',size=20)
plt.ylabel('成交数')
plt.xlabel('时间')
df1.成交时间.hist(bins=40,label='成交时间') #.成交时间
df1.挂牌时间.hist(bins=40,label='挂牌时间') #.挂牌时间
plt.legend(fontsize=12)
plt.tick_params(labelsize=13)
#plt.show()

df2019h1 = df1[(df1.成交时间>='20190101')&(df1.成交时间<='20190630')]    # 2019h1成交的房源信息
df2018h1 = df1[(df1.成交时间>='20180101')&(df1.成交时间<='20180630')]    # 2018h1成交的房源信息

df2019h1.to_csv('df2019h1.csv',index=False)
df2018h1.to_csv('df2018h1.csv',index=False)
