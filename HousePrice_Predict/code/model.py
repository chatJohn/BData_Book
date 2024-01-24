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

# 构建poi的dataframe用于后面存储相关poi信息
# 经纬度用于（1）poi数据获取（2）merge原数据集和poi数据集
df3 = pd.read_csv('df2019h1.csv').reset_index(drop=True)
df3[['经度','纬度']] = df3[['经度','纬度']].round(6) # 由于先前将经纬度转为了float，现在统一取后六位

# 首次运行，用locations存储经纬度集合，构建poi兴趣点df
locations = set((row[1],row[2]) for row in df3[['经度','纬度']].itertuples())
poi = pd.DataFrame(list(locations),columns=['经度','纬度'])


# %%time用于计算cell运行时间，必须放在首行
# 构建poi获取函数，基于百度地图开放品台api
def get_poi(row,search,radius):
    '''
    获取所需poi兴趣点的数量,如指定位置500m范围内地铁站的数量
    row：传入通过apply函数遍历数据集的每一行数据
    search：需要搜索的poi字段名称
    radius：范围m
    '''

    search = quote(search)
    lat_lng = str(row['纬度']) + ',' + str(row['经度'])
    radius = radius
    ak = 'gBDuxr5e5CMGG2aItaZE646qq4b98lsj'
    url = f"http://api.map.baidu.com/place/v2/search?query={search}&location={lat_lng}&radius={radius}&output=json&ak={ak}"
    try:
        result = requests.get(url)
    #time.sleep(0.2)     # 虽然I/O操作很慢，如有需要可考虑等待，以免超过服务的QPS限制。
        if result.status_code == 200:
            return len(result.json()['results'])
        else:
            return None  # 状态码非200则返回空值
    except Exception as e:  # except requests.exceptions.ConnectionError
        print('\n\n*** Requests.get(%s) gets wrong! ***\nThe program will try again later.\n\n' % url)
        return None  # 请求失败则返回空值

#调用百度地图api进行维度拓展
# 以下为地点检索获取poi数据的流程。
# 可选search列表：['地铁站','公交车站','三甲医院','小学','中学','幼儿园']
'''
i=0
for s in ['地铁站','公交车站','三甲医院','小学']: #,'公交车站','三甲医院','小学'
    for r in [500,1000,2000]:  #500,1000,2000
        poi[f'{s}_{r}m'] = poi.apply(get_poi,search=s,radius=r,axis=1)

poi.to_csv('poi.csv',index=False)
    #print('第'+i+'次收集成功')
    #i = i+1
# 由于地点检索的每日额度有限，可考虑先存储数据，分几天搜索poi数据
#poi.to_csv('poi.csv',index=False)
'''
df3 = df3.drop(['成交时间','调价','带看','关注','浏览','房屋年限','挂牌时间','梯户比例','交易权属',
                '建筑结构','挂牌价格','成交单价','成交周期','区域','子区域','套内面积','楼盘名称','房屋户型'],axis=1)
df3.info()
# poi经纬度注意也要取小数6位，以免影响merge。
poi = pd.read_csv('poi.csv')#.drop('楼盘名称',axis=1)
poi[['经度','纬度']] = poi[['经度','纬度']].round(6)
df_merge = pd.merge(df3,poi,how='left',on=['经度','纬度'])
df_merge.info()

# 异常值处理
def show_error(df,col,whis=1.5,show=False):
        '''
        显示上下限异常值数量，可选显示示例异常数据
        df：数据源
        col：字段名
        whis：默认1.5，对应1.5倍iqr
        show：是否显示示例异常数据
        '''
        iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
        upper_bound = df[col].quantile(0.75) + whis * iqr # 上界
        lower_bound = df[col].quantile(0.25) - whis * iqr # 下界
        # print(iqr,upper_bound,lower_bound)
        print('【',col,'】上界异常值总数：',df[col][df[col] > upper_bound].count())
        if show:
            print('异常值示例：\n',df[df[col] > upper_bound].head(5).T)
        print('【',col,'】下界异常值总数：',df[col][df[col] < lower_bound].count())
        if show:
            print('异常值示例：\n',df[df[col] < lower_bound].head(5).T)
        print('- - - - - - ')

def drop_error(df,col,whis=1.5):
        '''
        删除上下限异常值数量
        df：数据源
        col：字段名
        whis：默认1.5，对应1.5倍iqr
        '''
        iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
        upper_bound = df[col].quantile(0.75) + whis*iqr # 上界
        lower_bound = df[col].quantile(0.25) - whis*iqr # 下界
        data_del = df[col][(df[col] > upper_bound) | (df[col] < lower_bound)].count()
        data = df[(df[col] <= upper_bound) & (df[col] >= lower_bound)]
        print(f'【{col}】总剔除数据量：',data_del)
        return data.reset_index(drop=True)

# 处理空值
def fill_with_neighbor_or_mode(df,col):
    '''
    用相同经纬度的楼盘数据做填充，如果没有同类信息，则用整体众数填充。
    df：数据源
    col：字段名
    '''
    loc = df[df[col].isna()][['经度','纬度']]
    loclist = set((l[1],l[2]) for l in loc.itertuples())
    mode_number = df[col].mode()[0]
    # print(mode_number)
    for i in loclist:
        try:
            r = df[(df.经度 == i[0]) & (df.纬度 == i[1])][col].mode()[0]
            df.loc[(df.经度 == i[0]) & (df.纬度 == i[1]) & df[col].isna(),col] = r
        except Exception as e:
            df.loc[(df.经度 == i[0]) & (df.纬度 == i[1]) & df[col].isna(),col] = mode_number
    print(f'fill_with_neighbor_or_mode - {col} , done!')

def roomtype_get_dummies(df):
    '''
    房屋户型哑变量获取
    df：数据源
    '''
    roomtype = df.房屋户型.str.extract('(?P<室>\d+)室(?P<厅>\d+)厅(?P<厨>\d+)厨(?P<卫>\d+)卫').astype(np.int64)
    print('roomtype_get_dummies , done!')
    return pd.merge(df,roomtype,how='left',left_index=True,right_index=True).drop('房屋户型',axis=1)

# 房屋朝向哑变量获取
def orientation_get_dummies(df):
    '''
    房屋朝向哑变量获取
    df：数据源
    '''
    l = set()
    for i in df.房屋朝向.str.split(' '):
        l.update(i)
    d = pd.DataFrame(np.zeros((len(df),len(l)),dtype=np.int8),
                        columns=[f'房屋朝向_{i}' for i in l])
    df = df.join(d)
    for n in l:
        df.loc[df.房屋朝向.str.contains(n),f'房屋朝向_{n}'] = 1
    print('orientation_get_dummies , done!')
    return df.drop('房屋朝向',axis=1)

def dummies_getting(df,col):
    '''
    其他哑变量获取
    df：数据源
    col：字段名
    '''
    df = pd.get_dummies(df,columns=col)
    print(f'dummies_getting - {col} , done!')
    return df

df_merge.loc[df_merge.产权年限 == '未知','产权年限'] = np.nan
for i in ['户型结构','建筑类型','建成年代','配备电梯','房权所属','产权年限']:
    fill_with_neighbor_or_mode(df_merge,i)
df_merge.产权年限 = df_merge.产权年限.str[:2].astype(np.int8)

# 查看一下最终会保留的连续值字段异常值情况，注意这里whis是3
plt.figure(figsize=(16,9))
for n,i in enumerate(['成交价', '建筑面积','建成年代']):
    plt.subplot(1,4,n+1)
    plt.title(i)
    sns.boxplot(df_merge[i],orient='v',width=0.2,whis=3)
    plt.ylabel('')
for i in ['成交价', '建筑面积','建成年代']:
    show_error(df_merge,i,whis=3)

#处理异常值
for i in ['成交价', '建筑面积','建成年代']:
    df_merge = drop_error(df_merge,i,whis=3)    # 注意这里whis是3

# 房屋户型哑变量获取
#df_merge = roomtype_get_dummies(df_merge)

# 房屋朝向哑变量获取
#df_merge = orientation_get_dummies(df_merge)

# 其他哑变量获取
columns = ['户型结构','建筑类型','装修情况','配备电梯','房屋用途','房权所属']
df_merge = dummies_getting(df_merge,columns)

# 整体分布偏态情况减小
plt.figure(figsize=(16,9))
for n,i in enumerate(['成交价', '建筑面积','建成年代']):
    plt.subplot(4,1,n+1)
    plt.title(i)
    sns.distplot(df_merge[i])
    plt.ylabel('')

# 准备训练、测试集
X = df_merge.drop(['成交价'],axis=1)
y = df_merge['成交价']
xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2,random_state=42)    # random_state=42

# k折交叉拆分器 - 用于网格搜索
cv = KFold(n_splits=3,shuffle=True)

# 回归模型性能查看函数
def perfomance_reg(model,X,y,name=None):
    y_predict = model.predict(X)
    check = pd.DataFrame(y)
    check['y_predict'] = y_predict
    check['abs_err'] = abs(check['y_predict'] - check[y.name] )
    check['ape'] = check['abs_err'] / check[y.name]
    ape = check['ape'][check['ape']!=np.inf].mean()
    if name:
        print(name,':')
    print(f'mean squared error is: {mean_squared_error(y,y_predict)}')
    print(f'mean absolute error is: {mean_absolute_error(y,y_predict)}')
    print(f'R Squared is: {r2_score(y,y_predict)}')
    print(f'mean absolute percent error is: {ape}')
    print('- - - - - - ')

# 定义LassoCV（可自动通过Cross Validation搜索最佳正则系数alpha）
pipe_lassocv = Pipeline([
    ('poly',PolynomialFeatures(degree=2)),
    ('sc',StandardScaler()),
    ('pwt',PowerTransformer()),
    ('lasso_regr',LassoCV(
        # 待搜索alpha值如下：
        # alphas=(list(np.arange(1,10)*0.1)+list(np.arange(1,10))+list(np.arange(1,11)*10))
        # alpha搜索完后发现0.1最合适
        alphas = (list(np.arange(1,10)*0.1)+list(np.arange(1,10))+list(np.arange(1,11)*10)),
        cv=KFold(n_splits=3,shuffle=True),    # 配合搜索参数使用
        n_jobs=-1,verbose=1))
])
print(xtrain)
xtrain.replace(np.nan, 0, inplace=True)
ytrain.replace(np.nan, 0, inplace=True)
xtest.replace(np.nan, 0, inplace=True)
ytest.replace(np.nan, 0, inplace=True)
# 搜索参数并训练模型
pipe_lassocv.fit(xtrain,ytrain)
# 最佳参数组合
print('最佳alpha值：',pipe_lassocv.named_steps['lasso_regr'].alpha_)
# 训练集性能指标
perfomance_reg(pipe_lassocv,xtrain,ytrain,name='train')
# 测试集性能指标
perfomance_reg(pipe_lassocv,xtest,ytest,name='test')


# 定义RandomForestRegressor随机森林回归模型
rf_reg = RandomForestRegressor(criterion='mse',
                               n_jobs=-1,)    # random_state
# 参数设定
rf_grid_params = {'max_features':['auto'],    # ['auto',0.5,0.6,0.9] 带搜索参数
                    'max_depth':[8,9,10],    # [3,6,9]
                    'n_estimators':[500,800,1000]}
# 参数搜索
rf_gridsearch = GridSearchCV(rf_reg,rf_grid_params,cv=cv,
                               n_jobs=-1,scoring='neg_mean_squared_error',verbose=5,refit=True)
# 工作流管道
pipe_rf = Pipeline([
        ('sc',StandardScaler()),
        ('rf_grid',rf_gridsearch)
])

# 搜索参数并训练模型
pipe_rf.fit(xtrain,ytrain)
# 最佳参数组合
print('最佳alpha值：',pipe_rf.named_steps['rf_grid'].best_params_)
# 训练集性能指标
perfomance_reg(pipe_rf,xtrain,ytrain,name='train')
# 测试集性能指标
perfomance_reg(pipe_rf,xtest,ytest,name='test')

# xgboost模型
xgb_reg = xgb.XGBRegressor(objective='reg:linear',
                            n_job=-1,
                            booster='gbtree',
                            learning_rate=0.05)
# 参数设定
xgb_params = {'max_depth':[6,9],
             'subsample':[0.6,0.9,1],
             'colsample_bytree':[0.5,0.6],
             'reg_alpha':[0,0.05,0.1],
             'n_estimators':[750,1000]}
# 参数搜索
xgb_gridsearch = GridSearchCV(xgb_reg,xgb_params,cv=cv,n_jobs=-1,
                                verbose=10,refit=True)
# 工作流管道
pipe_xgb = Pipeline([
    ('sc',StandardScaler()),
    ('xgb_grid',xgb_gridsearch)
])

# 搜索参数并训练模型
pipe_xgb.fit(xtrain,ytrain)
# 最佳参数组合
print(pipe_xgb.named_steps['xgb_grid'].best_params_)
# 训练集性能指标
perfomance_reg(pipe_xgb,xtrain,ytrain,name='train')
# 测试集性能指标
perfomance_reg(pipe_xgb,xtest,ytest,name='test')

# 获取xgboost模型各特征重要程度数值
importances = pipe_xgb.named_steps['xgb_grid'].best_estimator_.feature_importances_
features = xtrain.columns
importance_table = pd.DataFrame({'features':features,'importances':importances})
importance_table.sort_values(by='importances',ascending=False)

im_median = importance_table.importances.median()
select_list = importance_table[importance_table.importances>=im_median].features.tolist()
select_list.extend(['ensemble_rf','成交价']) # '成交价'为方便后续拆分数据集；'ensemble_rf'为随机森林预测结果
print(select_list)

# 集成随机森林预测结果 并 根据重要特征列表筛选数据集
df_ensemble = df_merge.copy()
df_ensemble.replace(np.nan, 0, inplace=True)
df_ensemble['ensemble_rf'] = pipe_rf.predict(df_ensemble.drop('成交价',axis=1))
df_ensemble = df_ensemble.loc[:,select_list]

# 生成集成学习训练、验证集合
XX = df_ensemble.drop('成交价',axis=1)
yy = df_ensemble['成交价']
xxtrain,xxtest,yytrain,yytest = train_test_split(XX,yy,test_size=0.2,random_state=1)    # random_state=42

pipe_xgb_ensemble = Pipeline([
    ('sc',StandardScaler()),
    ('xgb_grid',xgb_gridsearch)
])

# 搜索参数并训练模型
pipe_xgb_ensemble.fit(xxtrain,yytrain)
# 最佳参数组合
print(pipe_xgb_ensemble.named_steps['xgb_grid'].best_params_)
# 训练集性能指标
perfomance_reg(pipe_xgb_ensemble,xxtrain,yytrain,name='train')
# 测试集性能指标
perfomance_reg(pipe_xgb_ensemble,xxtest,yytest,name='test')

# 由于我们没有新建网格搜索，原来的pipe_xgb被集成学习的pipe_xgb_ensemble覆盖。
# 利用joblib存储模型及相关数据
from sklearn.externals import joblib
joblib.dump({'model_name':'pipe_xgb_ensemble',
            'features_list':select_list,
            'model':pipe_xgb_ensemble,
            'data':[xxtrain,xxtest,yytrain,yytest]},
            'pipe_xgb_ensemble.pkl')