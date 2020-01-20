# -*- coding: utf-8 -*-
# @Author  : GJN
import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image = Image.open('Yu1.png')
code = pytesseract.image_to_string(image)


def show_boxes():
    from pytesseract import Output
    import cv2
    img = cv2.imread('Yu1.png')

    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)


def string_to_dict(image_string):
    s_list = image_string.split('\n')
    for i in range(len(s_list)):
        s_list[i] = '.'.join([s.strip() for s in s_list[i].split('.')])
        s_list[i] = '0h' if s_list[i] == 'Oh' else s_list[i]

    result = dict()
    if len(s_list) == 47:
        result['c'] = ['name'] + [s_list[7 * x + 6] for x in range(6)]
        result['FT3'] = [s_list[7 * x] for x in range(7)]
        result['FT4'] = [s_list[7 * x + 1] for x in range(7)]
        result['TT3'] = [s_list[7 * x + 2] for x in range(7)]
        result['TT4'] = [s_list[7 * x + 3] for x in range(7)]
        result['TSH'] = [s_list[7 * x + 4] for x in range(7)]

    return result


def std2(image_string):
    s_list = image_string.split('\n')
    result = [[]]
    j = 0
    for i in range(len(s_list)):
        if not s_list[i]:
            result.append([])
            j += 1
            continue
        s_list[i] = '.'.join([s.strip() for s in s_list[i].split('.')])
        s_list[i] = '0h' if s_list[i] == 'Oh' else s_list[i]
        result[j].append(s_list[i])
    return result

