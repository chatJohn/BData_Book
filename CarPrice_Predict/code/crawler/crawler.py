import json
import random
import re

import requests
import time

from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from CarInfo import CarInfo

# 请求头
headers = {
    'Referer': 'https://www.che168.com/',
    'User-Agent': ''
}
# 配置亮点dict（key为亮点id，value为亮点名称，最初由“亮点dict.json”文件加载而来）
highlights_dict = {}
# 用于请求车辆列表的user_agent
list_user_agent = ''

"""刷新user_agent"""
def refresh_headers():
    """
    刷新请求头
    :return: None
    """
    global headers
    headers['User-Agent'] = UserAgent().random

"""获取并解析当前网页的html"""
def get_html(url):
    """
    获取并解析当前网页的html
    :param url: 网址
    :return: 由BeautifulSoup解析后的html，若获取/解析失败则返回None
    """
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            # 判断网页编码
            charset = None
            html = response.text
            m = re.compile('<meta .*(http-equiv="?Content-Type"?.*)?charset="?([a-zA-Z0-9_-]+)"?', re.I).search(html)
            if m and m.lastindex == 2:
                charset = m.group(2).lower()
            response.encoding = charset
            response.close()
            return BeautifulSoup(html, 'html.parser')
        return None
    except Exception as e:
        my_print(e)
        return None

"""获取API返回的json"""""
def get_json(url):
    """
    获取当前网页的json
    :param url: 网址
    :return: json，若获取失败则返回None
    """
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            json_data = response.text
            response.close()
            return json.loads('{' + json_data.split('({')[1][:-1])
        return None
    except Exception as e:
        my_print(e)
        return None
def get_json_demo(filename):
    # 读取resources中名为filename的json文件
    with open('../resources/' + filename, 'r') as f:
        json_data = f.read()
    return json.loads(json_data)
def get_html_demo(filename):
    # 读取resources中的纯电车.html文件
    with open('../resources/' + filename, 'r') as f:
        html = f.read()
    return BeautifulSoup(html, 'html.parser')

"""加载亮点dict"""
def load_highlights_dict():
    """
    从“亮点dict.json”文件中加载亮点dict
    :return: None
    """
    global highlights_dict
    with open('亮点dict.json', 'r') as f:
        highlights_dict = json.loads(f.read())

"""刷新亮点dict"""
def refresh_highlights_dict(highlight):
    """
    刷新亮点dict
    :param highlight: 需要判断的亮点（dict）
    """
    global highlights_dict
    if highlight['optionid'] not in highlights_dict:
        highlights_dict[highlight['optionid']] = highlight['optionname']

"""保存highlights_dict到“亮点dict.json”文件中"""
def save_highlights_dict():
    """
    保存highlights_dict到“亮点dict.json”文件中
    :return: None
    """
    global highlights_dict
    with open('../resources/亮点dict.json', 'w') as f:
        f.write(json.dumps(highlights_dict))

"""判断str是否为浮点数"""
def is_float(str):
    """
    判断str是否为浮点数
    :param str: 需要判断的str
    :return: True/False
    """
    pattern = r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'
    if re.match(pattern, str):
        return True
    else:
        return False

"""获取当前有车辆的级别、车身颜色"""
def get_car_level_color(url):
    """
    获取当前有车辆的级别、车身颜色
    :param url: 车辆详情页的url（str）
    :return: 级别、车身颜色
    """
    # 获取当前车辆的html
    soup = get_html(url)
    if soup is None:
        return None, None
    # soup = get_html_demo('纯电车.html')
    # 获取信息页的参数表格
    base_info = soup.find_all('ul', class_='basic-item-ul')
    # 获取当前车辆的车身颜色
    color = base_info[2].find_all('li')[2].text[4:]
    # 车辆级别
    car_level = base_info[2].find_all('li')[1].text[4:]
    return car_level, color

"""获取车辆的热度、关注、咨询、搜索指数"""
def get_heat_score(seriesid, infoid):
    """
    获取车辆的热度、关注、咨询、搜索指数
    :param seriesid: seriesid（str）
    :param infoid:  infoid（str）
    :return: 热度、关注、咨询、搜索指数
    """
    # 构造url
    url = 'https://yccacheapigo.che168.com/api/carinfo/getheatrank?callback=getHeatRankCallback&_appid=2sc&seriesid={0}&infoid={1}'.format(seriesid, infoid)
    # 获取当前车辆指数的json
    json_data = get_json(url)
    if json_data is None:
        return None
    # json_data = get_json_demo('指数.json')
    # 获取热度、关注、咨询、搜索指数
    scores = [json_data['result']['search_score'],
              json_data['result']['focus_score'],
              json_data['result']['consults_score'],
              json_data['result']['heat_exponent']]
    return scores

"""获取车辆的配置亮点"""
def get_highlights(infoid):
    """
    获取车辆的配置亮点
    :param infoid: infoid（str）
    :return: 配置亮点（str）
    """
    # 构造url
    url = 'https://apipcmusc.che168.com/v1/car/getusedcaroptiondata?callback=getUsedCarOptionDataCallback&_appid=2sc.m&infoid={}'.format(infoid)
    # 获取当前车辆配置亮点的json
    json_data = get_json(url)
    if json_data is None:
        return None
    # json_data = get_json_demo('配置亮点.json')
    # 获取配置亮点
    highlights = json_data['result']
    hightlight_str = ''
    for highlight in highlights:
        optionid = str(highlight['optionid'])
        optionname = highlight['optionname']
        highlight_dict = {}
        highlight_dict['optionid'] = optionid
        highlight_dict['optionname'] = optionname
        # 刷新亮点dict
        refresh_highlights_dict(highlight_dict)
        # 将当前亮点添加到亮点str中
        hightlight_str += optionid + '、'
    # 去掉最后一个顿号
    hightlight_str = hightlight_str[:-1]
    return hightlight_str

"""获取车辆的详细参数信息"""
def get_detail_info(specid):
    """
    获取车辆的详细信息（参数）
    :param specid: specid（str）
    :return: 车辆详细信息（dict）
    """
    # 构造url
    url = 'https://cacheapigo.che168.com/CarProduct/GetParam.ashx?specid={}&callback=configTitle'.format(specid)
    # 获取当前车辆详细信息的json
    json_data = get_json(url)
    if json_data is None:
        return None
    # json_data = get_json_demo('参数.json')
    # 获取车辆详细信息
    detail_info = json_data['result']['paramtypeitems']
    # 获取所需要的参数名
    name_dict = CarInfo.getNameDict()
    # 将车辆详细信息转换为dict
    detail_info_dict = {}
    for detail in detail_info:
        table_param = detail['paramitems']
        for param in table_param:
            if param['name'] in name_dict:
                # 判断对应的value是否为数字
                if param['value'].isdigit():
                    detail_info_dict[param['name']] = int(param['value'])
                    continue
                if is_float(param['value']):
                    if param['name'] == '上市时间':
                        detail_info_dict[param['name']] = param['value']
                        continue
                    detail_info_dict[param['name']] = float(param['value'])
                    continue
                if param['value'] == '-':
                    detail_info_dict[param['name']] = 0
                    continue
                detail_info_dict[param['name']] = param['value']
    return detail_info_dict

"""保存csv文件的表头"""
def save_csv_head():
    """
    保存csv文件的表头
    """
    # 获取车辆信息的属性名
    name_dict = CarInfo.getNameDict()
    # 构造csv文件的表头
    csv_head = ''
    for key in name_dict:
        csv_head += key + ','
    csv_head = csv_head[:-1] + '\n'
    # 保存csv文件
    with open('car_info.csv', 'w') as f:
        f.write(csv_head)

"""保存csv文件的内容(没爬一条保存一次)"""
def save_csv_content(car_info):
    """
    保存csv文件的内容
    :param car_info: 车辆信息
    """
    # 获取车辆信息的属性名
    name_dict = CarInfo.getNameDict()
    # 构造csv文件的内容
    csv_content = ''
    for key in name_dict:
        csv_content += str(car_info.getValue(key)) + ','
    csv_content = csv_content[:-1] + '\n'
    csv_content = csv_content.encode('gbk', 'ignore').decode('gbk')
    # 保存csv文件
    with open('car_info.csv', 'a') as f:
        f.write(csv_content)

"""随机休眠(配合分级随机休眠)"""
def take_a_break(start, end, nead_my_print=True):
    """
    随机休眠start~end秒
    :param start: 最低休眠时间（int）
    :param end: 最高休眠时间（int）
    :param nead_my_print: 是否需要打印提示信息（bool）
    """
    # 随机休眠start~end秒
    sleep_time = start + (end - start) * random.random()
    if nead_my_print:
        my_print("=====================================================")
        my_print("稍等稍等，休息{}秒".format(sleep_time))
        my_print("=====================================================")
    time.sleep(sleep_time)

"""分级随机休眠"""
def take_a_brake_level(level = 0, nead_my_print=True):
    """
    分级随机休眠
    :param level: 休眠级别（int）
    :param nead_my_print: 是否需要打印提示信息（bool）
    """
    # 分级时段
    level_time = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (600, 1200)]
    # 随机休眠start~end秒
    take_a_break(level_time[level][0], level_time[level][1], nead_my_print)

"""打印信息并保存至log文件"""
def my_print(p_str):
    """
    打印信息并保存至log文件
    :param str: 需要打印的信息
    """
    # 获取当前时间
    now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print_str = str(now_time) + ': ' + str(p_str)
    print_str = print_str.encode('gbk', 'ignore').decode('gbk')
    # 保存至log文件
    with open('log.txt', 'a') as f:
        f.write(print_str + '\n')
    # 打印信息
    print(p_str)

"""爬虫主要函数"""
def run():
    list_url = 'https://www.che168.com/china/a0_0msdgscncgpi1ltocsp{}exx0/?pvareaid=102179#currengpostion'
    page_index = 1
    fail_count = 0
    fail_car_count = 0
    conti_success_count = 0
    # 记录已访问的车辆数
    count = 0
    while page_index < 101 and fail_count < 6:
        url = list_url.format(page_index)
        # 将请求头中的user_agent设置为请求车辆列表的user_agent
        headers['User-Agent'] = list_user_agent
        soup = get_html(url)
        # soup = get_html_demo('carlist.html')
        if soup is None:
            fail_count += 1
            my_print("获取第{}页失败，页面信息为空".format(page_index))
            conti_success_count = 0
            if fail_count > 5:
                my_print("连续失败超过5次，程序退出")
                break
            # 分级随机休眠
            take_a_brake_level(fail_count)
            continue
        # 获取当前页面的所以车辆的链接、名称、价格、上牌时间、行驶里程、所在地
        # 所以车辆信息li
        try:
            car_list = (soup.find_all('li', class_='cards-li list-photo-li') +
                        soup.find_all('li', class_='cards-li list-photo-li ') +
                        soup.find_all('li', class_='cards-li list-photo-li cxc-card'))
            my_print("************************成功获取第{}页页面************************".format(page_index))
        except Exception as e:
            my_print(e)
            my_print("获取第{}页失败，访问被ban了".format(page_index))
            fail_count += 1
            conti_success_count = 0
            if fail_count > 5:
                my_print("连续失败超过5次，程序退出")
                break
            # 分级随机休眠
            take_a_brake_level(fail_count)
            continue
        # 遍历所有车辆信息
        car_fail_count = 0  # 记录同一辆车连续失败的次数
        i = 0
        while i < len(car_list):
            get_specid = False
            get_succeed = False
            car = car_list[i]
            try:
                # 判断是否为车辆信息
                car_info_box = car.find('div', class_='cards-bottom')
                if car_info_box is None:
                    i += 1
                    continue
                # 读取该盒子中的信息
                car_info = CarInfo()
                # 车辆名称
                car_info.setValue('车辆名称', car['carname'].strip())
                # 车辆价格
                price = float(car['price'])
                car_info.setValue('车型报价', price)
                # 车辆上牌时间、行驶里程、所在地
                infos = car.find('p', class_='cards-unit').text.split('／')
                # 行驶里程(万公里)
                car_info.setValue('表显里程', float(car['milage']))
                # 上牌时间
                car_info.setValue('上牌时间', car['regdate'].replace('/', '.'))
                # 所在地
                car_info.setValue('所在地', infos[2])
                # seriesid
                seriesid = car['seriesid']
                car_info.setValue('car_seriesid', seriesid)
                # infoid
                infoid = car['infoid']
                car_info.setValue('car_infoid', infoid)
                # specid
                specid = car['specid']
                car_info.setValue('car_specid', specid)
                # dealerid
                dealerid = car['dealerid']
                # 车辆链接
                car_url = 'https://www.che168.com/dealer/{0}/{1}.html?pvareaid=100519&userpid=0&usercid=0&offertype=&offertag=0&activitycartype=0&fromsxmlist=0'.format(dealerid, car['infoid'])
                car_info.setValue('车源网址', car_url)

                # 刷新请求头
                refresh_headers()

                # 随机休眠1~3秒
                # take_a_break(1, 3, False)
                # 获取详情页中的车辆信息（级别、车身颜色）
                car_level, color = get_car_level_color(car_info.getValue('车源网址'))
                if car_level is None:
                    my_print("获取第{}辆信息失败，马上休息会，然后重试，级别和颜色详情页为空".format(count + 1))
                    car_fail_count += 1
                    conti_success_count = 0
                    if car_fail_count > 5:
                        my_print("第{}辆车连续失败超过5次，跳过此辆车".format(count + 1))
                        car_fail_count = 0
                        fail_count += 1
                        fail_car_count += 1
                        i += 1
                        count += 1
                        if fail_count > 5:
                            my_print("连续失败超过5次，程序退出")
                            # 分级随机休眠
                            take_a_brake_level(fail_count)
                            break
                        # 分级随机休眠
                        take_a_brake_level(fail_count)
                        continue
                    # 分级随机休眠
                    take_a_brake_level(car_fail_count)
                    continue
                if specid != '0':
                    get_specid = True
                car_info.setValue('车辆级别', car_level)
                car_info.setValue('车身颜色', color)

                # 获取车辆的热度、关注、咨询、搜索指数
                scores = get_heat_score(seriesid, infoid)
                if scores is None:
                    car_fail_count += 1
                    conti_success_count = 0
                    if car_fail_count > 5:
                        my_print("第{}辆车连续失败超过5次，跳过此辆车".format(count + 1))
                        car_fail_count = 0
                        fail_count += 1
                        fail_car_count += 1
                        i += 1
                        count += 1
                        if fail_count > 5:
                            my_print("连续失败超过5次，程序退出")
                            # 分级随机休眠
                            take_a_brake_level(fail_count)
                            break
                        # 分级随机休眠
                        take_a_brake_level(fail_count)
                        continue
                    my_print("获取第{}辆信息失败，马上休息会，然后重试，指数信息为空".format(count + 1))
                    # 分级随机休眠
                    take_a_brake_level(car_fail_count)
                    continue
                car_info.setValue('热度指数', scores[0])
                car_info.setValue('关注指数', scores[1])
                car_info.setValue('咨询指数', scores[2])
                car_info.setValue('搜索指数', scores[3])

                # 获取车辆的配置亮点
                highlights = get_highlights(infoid)
                if highlights is None:
                    car_fail_count += 1
                    conti_success_count = 0
                    if car_fail_count > 5:
                        my_print("第{}辆车连续失败超过5次，跳过此辆车".format(count + 1))
                        car_fail_count = 0
                        fail_count += 1
                        fail_car_count += 1
                        i += 1
                        count += 1
                        if fail_count > 5:
                            my_print("连续失败超过5次，程序退出")
                            # 分级随机休眠
                            take_a_brake_level(fail_count)
                            break
                        # 分级随机休眠
                        take_a_brake_level(fail_count)
                        continue
                    my_print("获取第{}辆信息失败，马上休息会，然后重试，亮点数据为空".format(count + 1))
                    # 分级随机休眠
                    take_a_brake_level(car_fail_count)
                    continue
                car_info.setValue('配置亮点', highlights)

                # 随机休眠1~3秒
                # take_a_break(1, 3, False)
                # 获取车辆的详细信息（参数）
                if get_specid:
                    detail_info_dict = get_detail_info(specid)
                    if detail_info_dict is None:
                        car_fail_count += 1
                        conti_success_count = 0
                        if car_fail_count > 5:
                            my_print("第{}辆车连续失败超过5次，跳过此辆车".format(count + 1))
                            car_fail_count = 0
                            fail_count += 1
                            fail_car_count += 1
                            i += 1
                            count += 1
                            if fail_count > 5:
                                my_print("连续失败超过5次，程序退出")
                                # 分级随机休眠
                                take_a_brake_level(fail_count)
                                break
                            # 分级随机休眠
                            take_a_brake_level(fail_count)
                            continue
                        my_print("获取第{}辆信息失败，马上休息会，然后重试，参数数据为空".format(count + 1))
                        # 分级随机休眠
                        take_a_brake_level(car_fail_count)
                        continue
                    for key in detail_info_dict:
                        car_info.setValue(key, detail_info_dict[key])
                get_succeed = True
            except Exception as e:
                my_print(e)
                if not get_succeed:
                    car_fail_count += 1
                    conti_success_count = 0
                    if car_fail_count > 5:
                        my_print("第{}辆车连续失败超过5次，跳过此辆车".format(count + 1))
                        car_fail_count = 0
                        fail_count += 1
                        fail_car_count += 1
                        i += 1
                        count += 1
                        if fail_count > 5:
                            my_print("连续失败超过5次，程序退出")
                            # 分级随机休眠
                            take_a_brake_level(fail_count)
                            break
                        # 分级随机休眠
                        take_a_brake_level(fail_count)
                        continue
                    my_print("获取第{}辆信息失败，马上休息会，然后重试，详细信息访问被ban了".format(count + 1))
                    # 分级随机休眠
                    take_a_brake_level(car_fail_count)
                    continue

            # 保存到csv文件
            try:
                save_csv_content(car_info)
            except Exception as e:
                my_print(e)
                my_print("保存第{}辆车信息失败".format(count + 1))
                fail_count += 1
                fail_car_count += 1
                i += 1
                count += 1
                if fail_count > 5:
                    my_print("连续失败超过5次，程序退出")
                    # 分级随机休眠
                    take_a_brake_level(fail_count)
                    break
                # 分级随机休眠
                take_a_brake_level(fail_count)
                continue

            # 连续失败次数清零
            fail_count = 0
            car_fail_count = 0

            # 分级随机休眠
            # take_a_brake_level(fail_count, False)

            count += 1
            i += 1
            conti_success_count += 1
            my_print("成功获取并保存{0}辆车辆信息，当前刚获取的车辆名称为：{1}*********当前已跳过{2}辆车".format(count, car_info.getValue('车辆名称'), fail_car_count))
            # 随机每获取3~8辆车辆信息，随机休息10~15秒
            sleep_count = random.randint(7, 8)
            if (count % sleep_count == 0 and count % 1000 != 0) or (conti_success_count % 5 == 0):
                conti_success_count = 0
                # 分级随机休眠，休眠级别为2，休眠时间为10-15秒
                take_a_brake_level(2)
            elif count % 1000 == 0:
                conti_success_count = 0
                # 分级随机休眠，休眠级别为6，休眠时间为600-1200秒
                take_a_brake_level(6)

        my_print("√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√成功获取第{}页所有数据√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√√".format(page_index))
        # 下一页
        page_index += 1
        # 分级随机休眠，休眠级别为3，休眠时间为15~20秒
        take_a_brake_level(3)

    my_print("成功获取{0}辆车辆信息，文件已保存到data文件夹下".format(count))

if __name__ == '__main__':
    # 加载亮点dict
    load_highlights_dict()

    # 保存csv文件的表头
    save_csv_head()

    # 随机获取请求车辆列表的user_agent
    list_user_agent = UserAgent().random

    # 运行爬虫
    run()

    # 保存亮点dict
    save_highlights_dict()
