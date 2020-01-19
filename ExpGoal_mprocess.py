# coding=utf-8 #
# Author GJN #
from multiprocessing import Pool
import time
from PIL import Image
import pytesseract
import cv2
from collections import Counter
import pandas as pd
import time
import colorsys
from matplotlib import pyplot as plt


def get_contours(picture):
    img = cv2.imread(picture)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ## 阈值分割
    ret, thresh = cv2.threshold(gray, 120, 255, 1)
    # s(thresh)

    ## 对二值图像执行膨胀操作
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    dilated = cv2.dilate(thresh, kernel)

    ## 轮廓提取，cv2.RETR_TREE表示建立层级结构
    image, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 提取小方格，其父轮廓都为0号轮廓
    # img = cv2.imread('xg_totals_round_13.jpg')
    result = []
    for i in range(len(hierarchy[0])):
        if hierarchy[0][i][3] == 0:
            if cv2.contourArea(contours[i]) > 1000:
                x, y, w, h = cv2.boundingRect(contours[i])
                # img = cv2.rectangle(img, (x - 1, y - 1), (x + w + 1, y + h + 1), (0, 0, 255), 2)
                result.append([x, y, w, h])
    df = pd.DataFrame(result)
    df2 = df.sort_values(by=[0, 1]).reset_index()
    return df2, gray


def read(args):
    index, gray_p = args
    return index, pytesseract.image_to_string(Image.fromarray(gray_p), config='-psm 10 sfz')


def get_table(picture):
    df2, gray = get_contours(picture)
    dflist = []
    for index, row in df2.iterrows():
        x, y, w, h = row[0], row[1], row[2], row[3]
        gray_p = gray[y: y + h, x: x + w]
        dflist.append([index, gray_p])

    start = time.clock()
    print('-' * 50)
    pool = Pool(5)
    result = pool.map(read, dflist)
    end = time.clock()
    print('-' * 50)
    print(end - start)
    print('-' * 50)

    nlist, nx = [], []
    for index, row in sorted(result):
        if index % 11 == 0 and index > 1:
            nlist.append(nx)
            nx = []
        nx.append(row)
    nlist.append(nx)

    ndic = {}
    for j, n in enumerate(nlist):
        ndic[j] = n
    dfn = pd.DataFrame(ndic)
    return dfn


if __name__ == '__main__':
    dfn = get_table('xg_totals_round_13.jpg')
















