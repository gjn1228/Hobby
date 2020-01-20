# -*- coding: utf-8 -*-
# @Author  : GJN
from PIL import Image
from io import BytesIO
import os

path = 'E:/BaiduYun/斩！赤红之瞳_第55回/'

for pic in os.listdir(path):
    # with open(path + pic, 'rb') as f:
    #     b = BytesIO()
    #     f.seek(15, 0)
    #
    #     b.write(f.read())
    #
    #     im = Image.open(b)
    #     im.load()
    im = Image.open(path + pic)
    if im.mode == "RGBA":
        im.load()  # required for png.split()
        background = Image.new("RGB", im.size, (255, 255, 255))
        background.paste(im, mask=im.split()[3])  # 3 is the alpha channel
        im = background
    im.save(path + pic.replace('.webp', '.jpg'))


















