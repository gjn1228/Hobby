# -*- coding: utf-8 -*-
# @Author  : GJN
import requests

m_path = 'https://1252348479.vod2.myqcloud.com/92e0c654vodtransgzp1252348479/5508aca75285890795433152330/voddrm.token.59ed8da677ddb6a1106578dfce96872c8eb9455f894fc6bd14c2f587365679604b56106a73d2eeff5155837c305f54f4a980b22adc34ee7b04dfb698058051b0ccebc561ad12db3db8816f008b5ab88fc398e68632794e7642b6b7e6adc7fdb52ba1be9e83a68595daafeefd9a2a48373b2dca9c51b628bf1564408914cc3bf360fc34691d248915ceb1fdf8146d108717bf9d575d89e6c3a46087ff08f632413de0a5a3157c1e5cbc44e9c890430913.v.f35141.m3u8?t=5ed879de&us=UwTxi324D6&sign=02a545beec43bbad2ad03c1d9329c056	'
p2 = m_path.split('?t=')[0]
pre_path = '/'.join(m_path.split('/')[:5])
m_folder = r'E:\丁香园\\'
m_name = 'test1'
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36'


def get_ts_urls(m3u8_url, base_url, name, folder):
    r = requests.get(m3u8_url, headers={'User-Agent': user_agent})
    with open(folder + name + '.m3u8', 'wb') as f:
        f.write(r.content)
    urls = []
    with open(folder + name + '.m3u8', "r") as file:
        lines = file.readlines()

    for line in lines:
        if '.ts?' in line:
            urls.append(base_url+line.strip("\n"))
    return urls


u1 = get_ts_urls(m_path, pre_path, m_name, m_folder)


# m_path = 'https://1252348479.vod2.myqcloud.com/92e0c654vodtransgzp1252348479/5508aca75285890795433152330/voddrm.token.05da2057ff92b33eaae3a19286c56d8d7c6c25ade185d5e5d1cd13e82489bcd98b3c73745fff3fa3b5baeb4b62fadf7bfb9f01bcba437ea421309c318b82f8e0b6be34c2d1a87530267e6e3d6689c4d33f185183c549fbff9cf37de1269388ad235593397bd203c1cdfb62d48ce7fa660859cb91afb4d8dbe138734bc06c7637da3fb46efb5c4312c6960f6f86a1530574c08dea6507fb61acbe74377d008989bc1613cd05d1f77af1dd5b91917e49dc.v.f35141.m3u8?t=5ed8a6f1&us=lTYaSI1Ug6&sign=b9e8d859e85f28d02ab97b5fc66f4601'
# m_name = 'test1'
#
# pre_path = '/'.join(m_path.split('/')[:5]) + '/'
#
# u1 = get_ts_urls(m_path, pre_path, m_name)
#
# ts_download(u1, 'test1', r'E:\丁香园\test1\\')
#
# dir_path = r'E:\丁香园\test1\\'
# files = []
# for file in os.listdir(dir_path):
#     if file.endswith('.ts'):
#         files.append(dir_path + file)
# files.sort(key=lambda x: int(x.split('-')[-1].split('.ts')[0]))
# with open(dir_path + m_name + '.ts', 'wb+') as fw:
#     for f in files:
#         fw.write(open(f, 'rb').read())







