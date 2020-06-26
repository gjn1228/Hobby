# -*- coding: utf-8 -*-
# @Author  : GJN
from selenium import webdriver
import time
import datetime as dt
import os
import pymongo

# 初始化
client = pymongo.MongoClient(host='localhost', port=27017)
db = client['admin']
mongodb = db['dxy_test']

temp_folder = r'E:\丁香园\temp\\'


def rename_newest(t0, name, target_folder=''):
    result = 0
    target_folder = temp_folder if len(target_folder) == 0 else target_folder
    for f in os.listdir(temp_folder):
        if dt.datetime.fromtimestamp(os.stat(temp_folder + f).st_ctime) > t0:
            os.rename(temp_folder + f, target_folder + name)
            result = 1
            break
    if result == 1:
        return target_folder + name
    else:
        time.sleep(1)
        rename_newest(t0, name, target_folder)


def downloads_done():
    for i in os.listdir(temp_folder):
        if ".crdownload" in i:
            time.sleep(0.5)
            downloads_done()
        return 'Downloads Done'


def get_ts_urls(m3u8_url, base_url, name):
    urls = []
    t0 = dt.datetime.now()

    driver.get(m3u8_url)
    file_path = rename_newest(t0, name + t0.strftime('%Y%m%d%H%M%S') + '.m3u8')

    with open(file_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        if '.ts?' in line:
            urls.append(base_url+line.strip("\n"))
    return urls


chromeOptions = webdriver.ChromeOptions()
chromeOptions.add_argument("--proxy-server=http://127.0.0.1:8080")
prefs = {
    'profile.default_content_settings.popups': 0,
    'download.default_directory': temp_folder
}
chromeOptions.add_experimental_option('prefs', prefs)

driver = webdriver.Chrome(executable_path=r'D:\PycharmProjects\First\chromedriver', chrome_options=chromeOptions)
# driver.get('https://class.dxy.cn/clazz/course/413')
driver.get('https://class.dxy.cn/clazz/course/54?sr=22&nm=sylxhyzx&pd=class')
input('扫码')


def get_pdic():
    # 视频列表
    p_list = driver.find_elements_by_xpath('//*[@id="root"]/div/div[2]/div[1]/div/div[2]/div[2]/div[2]/div[2]/div/div[2]/p[1]')
    p_dic = {x + 1: y.text for x, y in enumerate(p_list)}
    return p_dic


# video_folder = r'E:\丁香园\临床预测模型从入门到精通\\'
video_folder = r'E:\丁香园\医学统计学从入门到精通\\'


def ts_download(urls, name, target_folder):
    # i = 1

    # def move_file():
    #     global i
    #
    #     print(rename_newest(t0, '%s-%s.ts' % (name, index + 1), target_folder=target_folder),
    #           ' %s/%s--%s' % (index + 1, len(urls), time.clock() - tx))

    tx = time.clock()
    assert isinstance(urls, list)

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    for index, url in enumerate(urls):
        # t0 = dt.datetime.now()
        f1 = '.'.join(url.split('/')[-1].split('.')[:2]) + '.ts'
        i = 0
        while f1 in os.listdir(temp_folder):
            time.sleep(0.1)
            i += 1
            if i > 10:
                raise IndexError('Copy Error')
        driver.get(url)
        i = 0
        while f1 not in os.listdir(temp_folder):
            time.sleep(0.3)
            i += 1
            if i > 50:
                raise IndexError('Download Error')
        os.rename(temp_folder + f1, target_folder + '%s-%s.ts' % (name, index + 1))
        print(target_folder + '%s-%s.ts' % (name, index + 1), ' %s/%s--%s' % (index + 1, len(urls), time.clock() - tx))


def download_video(p_index, video_folder, p_dic):
    p_name = p_dic[p_index]
    data = mongodb.find_one({'courseHourName': p_name})
    if data.get('is_download') == 1:
        print('Already Download: %s.%s' % (p_index, p_name))
        return
    pre_path = '/'.join(data['m_url'].split('/')[:5]) + '/'
    u1 = []
    for line in data['ts'].split('\n'):
        if '.ts?' in line:
            u1.append(pre_path+line.strip("\n"))
    ts_download(u1, p_name[:10], temp_folder)
    files = []
    for file in os.listdir(temp_folder):
        if file.endswith('.ts'):
            files.append(temp_folder + file)
    files.sort(key=lambda x: int(x.split('-')[-1].split('.ts')[0]))
    with open(video_folder + str(p_index) + '.' + p_name + '.ts', 'wb+') as fw:
        for f in files:
            fw.write(open(f, 'rb').read())
    for f in files:
        os.remove(f)
    mongodb.update_one({'courseHourName': p_name}, {'$set': {'is_download': 1}})


# p_index = 1


def video_main(p_index, video_folder, p_dic):
    p_name = p_dic[p_index]
    error = 0
    while not mongodb.find_one({'courseHourName': p_name}):
        error += 1
        if error > 10:
            print('Click TimeOut')
            break
        driver.find_element_by_xpath('//*[@id="root"]/div/div[2]/div[1]/div/div[2]/div[2]/div[2]/div[2]/div[%s]' % p_index).click()
        driver.implicitly_wait(2)
        driver.find_element_by_xpath('//*[@id="vjs_video_3"]/div[4]/button[1]').click()
        driver.implicitly_wait(1)
    download_video(p_index=p_index, video_folder=video_folder, p_dic=p_dic)


# download_video(4, video_folder=video_folder, p_dic=p_dic)
# donwload_video(6, video_folder=video_folder, p_dic=p_dic)


def download_video2(p_index, video_folder, p_dic, course_name, diff=0):
    p_name = p_dic[p_index]
    data = mongodb.find_one({'c': p_index - diff, 'course_name': course_name})
    if data.get('is_download') == 1:
        print('Already Download: %s.%s' % (p_index, p_name))
        return
    pre_path = '/'.join(data['m_url'].split('/')[:5]) + '/'
    u1 = []
    for line in data['ts'].split('\n'):
        if '.ts?' in line:
            u1.append(pre_path+line.strip("\n"))
    ts_download(u1, p_name[:10], temp_folder)
    files = []
    for file in os.listdir(temp_folder):
        if file.endswith('.ts'):
            files.append(temp_folder + file)
    files.sort(key=lambda x: int(x.split('-')[-1].split('.ts')[0]))
    with open(video_folder + str(p_index) + '.' + p_name + '.ts', 'wb+') as fw:
        for f in files:
            fw.write(open(f, 'rb').read())
    for f in files:
        os.remove(f)
    mongodb.update_one({'c': p_index - 4}, {'$set': {'is_download': 1}})


p_dic = get_pdic()
# for i in range(19, 42):
#     download_video2(i, video_folder=video_folder, p_dic=p_dic, diff=2)
course_name = driver.find_element_by_xpath('//*[@id="root"]/div/div[2]/div[1]/div/div[2]/div[1]/p').text
download_video2(4, video_folder=video_folder, p_dic=p_dic, diff=-3, course_name=course_name)



