# -*- coding: utf-8 -*-
import requests,time,re,csv,os#,pandas_profiling
from bs4 import BeautifulSoup as BS
from multiprocessing import Pool
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from urllib.request import quote
#import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PowerTransformer,PolynomialFeatures
from sklearn.linear_model import LinearRegression,LassoCV,LogisticRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import KFold,train_test_split,StratifiedKFold,GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,accuracy_score, \
                            precision_score,recall_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def url_analysis(url, h):
    '''
    访问网页，返回已解析的网页代码
    url：网址
    h：访问网页的头部信息
    '''
    try:
        r = requests.get(url, headers=h, timeout=10)
        #print(r.status_code)
        # print(url)
        soup = BS(r.text, 'lxml')
        if r.status_code == 200:
            return soup  # 请求成功，返回已解析网页代码
        else:
            return None  # 状态码非200则返回空值
    except Exception as e:  # except requests.exceptions.ConnectionError
        print('\n\n*** Requests.get(%s) gets wrong! ***\nThe program will try again later.\n\n' % url)
        return None  # 请求失败则返回空值

def get_urlist(soup):
    '''
    从已解析的网页中返回该页所有二级网址列表,针对不同对象需要修改相应查找方式
    soup:已解析的网页代码
    '''
    tags = soup.find('ul',class_="listContent").find_all('a',class_="img")
    urlist = [i['href'] for i in tags]
    return urlist

def get_info(soup):
    '''
    提取目标网页相关信息，针对不同对象需要修改相应查找方式
    soup：已解析的网页代码
    '''
    # 楼盘名称,经度,纬度,成交时间,成交价,成交单价,区域,子区域
    houseinfo1 = []
    # 为对每条代码执行异常处理，将代码组成代码列表
    houseinfo1_code = [
        r"""soup.find('div',class_="wrapper").text.split(' ')[0]""",
        r"""re.search(r"resblockPosition:'(.+),(.+)'",soup.text).group(1)""",
        r"""re.search(r"resblockPosition:'(.+),(.+)'",soup.text).group(2)""",
        r"""soup.find('div',class_="wrapper").find('span').text.split(' ')[0]""",
        r"""soup.find('div',class_="price").find('i').text""",
        r"""soup.find('div',class_="price").find('b').text""",
        r"""soup.find('div',class_="deal-bread").find_all('a')[2].text""",
        r"""soup.find('div',class_="deal-bread").find_all('a')[3].text""",
    ]
    for i in houseinfo1_code:
        try:
            houseinfo1.append(eval(i))
        except Exception as e:
            houseinfo1.append(np.nan)
            print('houseinfo1 '+str(i)+' got wrong，请检查网址:'+soup.find('link',rel='canonical')['href']) #用于检查网址

    # 挂牌价格,成交周期,调价,带看,关注,浏览
    houseinfo2 = []
    for i in range(6):
        try:
            houseinfo2.append(soup.find('div',class_="msg").find_all('label')[i].text)
        except Exception as e:
            houseinfo2.append(np.nan)
            #print('houseinfo2 '+str(i)+' got wrong，请检查网址:'+soup.find('link',rel='canonical')['href'])    #用于检查网址

    # 房屋户型,所在楼层,建筑面积,户型结构,套内面积,建筑类型,房屋朝向,建成年代,装修情况,建筑结构
    # 供暖方式,梯户比例,产权年限,配备电梯,链家编号,交易权属,挂牌时间,房屋用途,房屋年限,房权所属
    houseinfo3 = []
    for i in range(20):
        try:
            houseinfo3.append(soup.find('div',class_="newwrap baseinform").find_all('li')[i].text[4:].strip())
        except Exception as e:
            houseinfo3.append(np.nan)
            #print('houseinfo2 '+str(i)+' got wrong，请检查网址:'+soup.find('link',rel='canonical')['href'])    #用于检查网址

    info = [*houseinfo1,*houseinfo2,*houseinfo3]
    return info

def get_missed(missed,datam,h,i):
    '''
    尝试取回访问异常的网址数据，分为获取网址列表和获取信息两种模式
    missed：访问异常的网址列表
    datam：代入用于追加网址列表或信息的集合变量
    h：访问网页的头部信息
    i：i=1时为获取网址列表，i=2时为获取信息
    '''
    if missed != []:
        for u in missed:
            print('////// Getting back: '+u)
            soup = url_analysis(u,h)
            if soup != None:    #判断请求网页是否成功
                if i == 1:
                    urlist = get_urlist(soup)
                    datam.extend(urlist)
                elif i == 2:
                    info = get_info(soup)
                    datam.append(info)
                print('////// Getting back succeeded!'+u)
            else:
                print('\n*** Failed to get '+u+' back! ***')

def data_write(data,col,i=None):
    '''
    将数据写入文件，col为None时文件首行不写入列名，只写入数据
    data: 数据列表
    col: 特征列表
    i: 单进程时默认为None，多进程时应代入当前进程号。
    '''
    t = time.strftime("%Y%m%d %H%M%S", time.localtime())

    if i == None:   # 非异步时：
        with open(t+'all_data.csv','w',newline='',encoding='utf-8') as f:
            f.write(col+'\n')
            writer = csv.writer(f)
            for row in data:
                writer.writerow(row)
    else:  # 异步时：
        if col == None:     #判断是否需要在文件首行加入列名，再写入数据
            with open(str(i)+'.csv','w',newline='',encoding='utf-8') as f:
                writer = csv.writer(f)
                for row in data:
                    writer.writerow(row)
        else:    # 一般情况不使用
            with open(str(i)+'.csv','w',newline='',encoding='utf-8') as f:
                f.write(col+'\n')
                writer = csv.writer(f)
                for row in data:
                    writer.writerow(row)

def gather_data(pcs,col):
    '''
    多进程时，用于汇集所有数据并写入一个文件中
    pcs：总进程数
    col: 特征列表
    '''
    t = time.strftime("%Y%m%d %H%M%S", time.localtime())

    with open(t+'all_data.csv','w',encoding='utf-8') as f:
        f.write(col+'\n')

    for n in range(0,pcs):
        with open(str(n)+'.csv','r',encoding='utf-8') as fr:
            content = fr.read()
            with open(t+'all_data.csv','a',encoding='utf-8') as fw:
                fw.write(content)

def async_main(url,h,start,end,col,pcs):
    '''
    异步调用main()函数。汇总数据路径为默认，修改请参考gather_data()
    url:网址范例
    h：访问网页的头部信息
    start：起始页码，比如第1页则输入1
    end：终止页码，比如第10页则输入10
    col：特征列表
    pcs：任务总进程数
    '''
    part = int((end - start + 1)/pcs)   #part为每个进程的任务量（最后一个进程数量可能不同）
    p = Pool(50)    #允许同时运行最大进程数

    for i in range(0,pcs):
        if i < pcs - 1:     #非末位进程始终页码分配
            p.apply_async(main,args=(url,h,start+part*i,start+part*(i+1)-1,col,i))
        else:   #末位进程始终页码分配
            p.apply_async(main,args=(url,h,start+part*i,end,col,i))

    p.close()   #不再增加进程
    p.join()    #等待所有子进程结束

    gather_data(pcs,col)

def main(url,h,start,end,col,i=None):
    '''
    从一级网址提取此进程负责的所有网址列表，解析网址列表得到所需信息，返回信息列表,
    输出的数据文件路径为默认，若需修改请参考data_write()
    url:网址范例
    h：访问网页的头部信息
    start：起始页码，比如第1页则输入1
    end：终止页码，比如第10页则输入10
    col：特征列表
    i：进程号，默认为None表示同步运行
    '''
    #print('('+str(os.getpid())+')')

    # 1.得到此进程负责的所有网址列表
    list_all = []
    missed1 = []
    for n in range(start,end+1):                        #start-起始页，end-终止页
        soup_url = url_analysis(url+str(n)+'/',h)       #解析单页一级网址
        if soup_url != None:                            #判断请求网页是否成功
            houses_urlist = get_urlist(soup_url)        #生成单页所有商家网址列表
            list_all.extend(houses_urlist)              #生成该进程负责的所有网址列表
        else:
            missed1.append(url+str(n)+'/')
            continue
    get_missed(missed1,list_all,h,1)                    #最后若missed1非空，则再次尝试解析失败的网址

    # 2.解析网址列表，得到所需信息
    data_all = []
    missed2 = []
    for u in list_all:
        soup_house = url_analysis(u,h)
        if soup_house != None:                  #判断请求网页是否成功
            house_info = get_info(soup_house)
            data_all.append(house_info)
        else:
            missed2.append(u)
            continue
    get_missed(missed2,data_all,h,2)               #最后若missed2非空，则再次尝试解析失败的网址

    # 3.写入数据
    if i == None:    #判断是否异步调用
        data_write(data_all,col)
    else:
        data_write(data_all,None,i)
        #print('\n*** process '+str(i)+' done! ***\n')

if __name__ == '__main__':
    '''
    爬取数据：需要异步用async_main()，同步则用main()
    '''
    star = time.time()
    #print(os.getpid())

    #深圳各区成交二手房网址列表
    urllist = ["https://cq.lianjia.com/chengjiao/jiangbei/pg",
                "https://cq.lianjia.com/chengjiao/yubei/pg",
                "https://cq.lianjia.com/chengjiao/nanan/pg",
                "https://cq.lianjia.com/chengjiao/banan/pg",
                "https://cq.lianjia.com/chengjiao/shapingba/pg",
                "https://cq.lianjia.com/chengjiao/jiulongpo/pg",
                "https://cq.lianjia.com/chengjiao/yuzhong/pg",
                "https://cq.lianjia.com/chengjiao/dadukou/pg",
    ]
    '''
                "https://cq.lianjia.com/chengjiao/nanan/pg",
                "https://cq.lianjia.com/chengjiao/banan/pg",
                "https://cq.lianjia.com/chengjiao/shapingba/pg",
                "https://cq.lianjia.com/chengjiao/jiulongpo/pg",
                "https://cq.lianjia.com/chengjiao/yuzhong/pg",
                "https://cq.lianjia.com/chengjiao/beibei/pg",
                "https://cq.lianjia.com/chengjiao/jiangjing/pg",
                "https://cq.lianjia.com/chengjiao/dadukou/pg",
    '''
    #特征列表
    columns = ('楼盘名称,经度,纬度,成交时间,成交价,成交单价,区域,子区域,挂牌价格,成交周期,调价,带看,关注,浏览,'
                '房屋户型,所在楼层,建筑面积,户型结构,套内面积,建筑类型,房屋朝向,建成年代,装修情况,建筑结构,'
                '供暖方式,梯户比例,产权年限,配备电梯,链家编号,交易权属,挂牌时间,房屋用途,房屋年限,房权所属')
    h = {
        'Host':'cq.lianjia.com',
        'Referer':'https://cq.lianjia.com/chengjiao/',
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.79 Safari/537.36',
    }

    for url in urllist:
        #main(url,h,1,1,columns,i=None)     #同步爬取数据
        async_main(url,h,1,100,columns,50)     #异步爬取数据,目前设置不超过50进程。
        print('\nYYY  %s  YYY\n' % url)

    #删除多进程留下的数据文件
    for i in range(50):
        path = str(i)+'.csv'
        if os.path.exists(path):  # 如果文件存在则删除
            os.remove(path)

    end = time.time()
    print('\n\nUsed %ss' % (end-star))
    #  将各区成交二手房数据连接成单独csv文件
    files = [x for x in os.listdir('.') if os.path.isfile(x) and os.path.splitext(x)[1]=='.csv']
    date = [pd.read_csv(v) for v in files]
    df = pd.concat([i for i in date])
    df.to_csv('20191217data.csv')
