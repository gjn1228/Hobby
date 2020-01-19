# coding=utf-8 #
# Author GJN #
from PIL import Image
import pytesseract
import cv2
import numpy as np
from matplotlib import pyplot as plt
import io
import skimage
from collections import Counter
import pandas as pd
import time
import colorsys


def nouse():
    def show(img):
        plt.imshow(img)
        plt.show()

    # img = Image.open('xg_totals_round_13.jpg')
    # text = pytesseract.image_to_string(Image.open('xg_totals_round_13.jpg'),lang="equ+eng")

    # # 转灰度图
    # im = img.convert('L')
    # 二值化
    # im = Image.open('xg_totals_round_13.jpg').convert('1')
    # img = np.array(im)

    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.axis("off")
    # cv2.title("Input Image")
    # cv2.imshow(gimg, cmap="gray")
    # cv2.show()

    # img = io.imread('xg_totals_round_13.jpg', True)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(gimg, kernel)
    dilated = cv2.dilate(gimg, kernel)

    cv2.imshow('1', dilated)

    def border(img):
        # 构造一个3×3的结构元素
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilate = cv2.dilate(img, element)
        erode = cv2.erode(img, element)

        # 将两幅图像相减获得边，第一个参数是膨胀后的图像，第二个参数是腐蚀后的图像
        result = cv2.absdiff(dilate, erode)

        # 上面得到的结果是灰度图，将其二值化以便更清楚的观察结果
        retval, result = cv2.threshold(result, 40, 255, cv2.THRESH_BINARY)
        # 反色，即对二值图每个像素取反
        result = cv2.bitwise_not(result)
        # 显示图像
        cv2.imshow("result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def cross(img):
        diamond = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        diamond[0, 0] = 0
        diamond[0, 1] = 0
        diamond[1, 0] = 0
        diamond[4, 4] = 0
        diamond[4, 3] = 0
        diamond[3, 4] = 0
        diamond[4, 0] = 0
        diamond[4, 1] = 0
        diamond[3, 0] = 0
        diamond[0, 3] = 0
        diamond[0, 4] = 0
        diamond[1, 4] = 0
        square = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        x = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        # 使用cross膨胀图像
        result1 = cv2.dilate(img, cross)
        # 使用菱形腐蚀图像
        result1 = cv2.erode(result1, diamond)

        # 使用X膨胀原图像
        result2 = cv2.dilate(img, x)
        # 使用方形腐蚀图像
        result2 = cv2.erode(result2, square)

        # 将两幅闭运算的图像相减获得角
        result = cv2.absdiff(result2, result1)
        # 使用阈值获得二值图
        retval, result = cv2.threshold(result, 40, 255, cv2.THRESH_BINARY)

        cv2.imshow("result", result)
        data = []
        xs = []
        ys = []
        for j in range(int(result.size / result.shape[2])):
            y = j / result.shape[0]
            x = j % result.shape[0]

            if (result[x, y] - (255, 255, 255)).any():
                data.append([x, y])
                xs.append(x)
                ys.append(y)

        print(Counter(xs).most_common(10))
        print(Counter(ys).most_common(10))

    def blur():
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
        gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)

        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)
        blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
        (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)

    def jiugongge():
        boxes = []
        for i in range(len(hierarchy[0])):
            if hierarchy[0][i][3] == 0:
                boxes.append(hierarchy[0][i])

        ## 提取数字，其父轮廓都存在子轮廓
        number_boxes = []
        ix = img.copy()
        for j in range(len(boxes)):
            if boxes[j][2] != -1:
                ix = cv2.drawContours(ix, contours[boxes[j][0]], 0, (0, 0, 255), 2)
                # number_boxes.append(boxes[j])
                # x, y, w, h = cv2.boundingRect(contours[boxes[j][1]])
                # number_boxes.append([x, y, w, h])
                # img = cv2.rectangle(img, (x - 1, y - 1), (x + w + 1, y + h + 1), (0, 0, 255), 2)


def s(img):
    cv2.destroyAllWindows()
    cv2.imshow('1', img)


img = cv2.imread('xg_totals_round_13.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
## 阈值分割
ret, thresh = cv2.threshold(gray, 120, 255, 1)
s(thresh)
# s(thresh)

## 对二值图像执行膨胀操作
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
dilated = cv2.dilate(thresh, kernel)

## 轮廓提取，cv2.RETR_TREE表示建立层级结构
image, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 提取小方格，其父轮廓都为0号轮廓
img = cv2.imread('xg_totals_round_13.jpg')
result = []
for i in range(len(hierarchy[0])):
    if hierarchy[0][i][3] == 0:
        if cv2.contourArea(contours[i]) > 1000:
            x, y, w, h = cv2.boundingRect(contours[i])
            img = cv2.rectangle(img, (x - 1, y - 1), (x + w + 1, y + h + 1), (0, 0, 255), 2)
            result.append([x, y, w, h])
df = pd.DataFrame(result)
df2 = df.sort_values(by=[0, 1]).reset_index()
# img1 = img[314:342, 661:743]
# pytesseract.image_to_string(Image.fromarray(img1))
# g2 = gray[314:342, 661:743]
# pytesseract.image_to_string(Image.fromarray(g2))

data = []
i = 1
all_value = []

path = r'E:\Company\Tools\Exp_Goal\\'

start = time.clock()
# new_dic = {}
nlist, nx = [], []
for index, row in df2.iterrows():
    if index % 11 == 0 and index > 1:
        nlist.append(nx)
        nx = []
    x, y, w, h = row[0], row[1], row[2], row[3]
    gray_p = gray[y: y+h, x: x+w]

    # if index // 11 == 1 or index // 11 == 5:
    #     # gray_p = cv2.threshold(gray_p, 180, 255, 0)[1]
    #     gray_p = cv2.adaptiveThreshold(gray_p, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 7)
    nx.append(pytesseract.image_to_string(Image.fromarray(gray_p), config='-psm 10 sfz'))

    # new_dic[index] = pytesseract.image_to_string(Image.fromarray(gray_p), config='-psm 10 sfz')
    # _, thresh_p = cv2.threshold(gray_p, 180, 255, 0)
    # cv2.imwrite(path + '%s.jpg' % (index + 100), thresh_p)
nlist.append(nx)
ndic = {}
for j, n in enumerate(nlist):
    ndic[j] = n
dfn = pd.DataFrame(ndic)
end = time.clock()
print(end - start)

j1 = cv2.imread(path + '%s.jpg' % 15)

_, thresh_p = cv2.threshold(j1, 180, 255, 0)
s(thresh_p)
value = pytesseract.image_to_string(Image.fromarray(thresh_p), config='-psm 10 sfz')
print(value)


def main_color(j1):
    image = Image.fromarray(j1)

    max_score, dominant_color = 0, 0
    for count, (r, g, b) in image.getcolors(image.size[0] * image.size[1]):

        saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]

        y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)

        y = (y - 16.0) / (235 - 16)

        score = (saturation + 0.1) * count

        if score > max_score:
            max_score = score
            dominant_color = (r, g, b)

    return dominant_color