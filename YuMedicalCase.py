# -*- coding: utf-8 -*-
# @Author  : GJN
import pytesseract
from PIL import Image
import os
import cv2
from pytesseract import Output
import pandas as pd
import xml.etree.ElementTree as xmlDoc
import time
import pypinyin
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

# image = Image.open('Yu1.png')
# code = pytesseract.image_to_string(image)


def count_time(func_name=''):
    def timeit(func):
        def wrapper(*arg, **kw):
            start = time.perf_counter()
            print('%s Running...' % func_name if func_name else '%s Running...' % func.__name__)
            df = func(*arg, **kw)
            end = time.perf_counter()
            print('%s Done in %ss' % (func_name, end - start) if func_name else '%s Done in %ss' % (func.__name__, end - start))
            return df
        return wrapper
    return timeit


def read_pic():

    def show_boxes():
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

    path = r'E:\zxy\tif\\'
    # path = r'E:\zxy\金亮'
    os.chdir(path)
    i1 = Image.open('白立新_F.tif')
    c1 = pytesseract.image_to_string(i1, lang='chi_sim')
    c11 = pytesseract.image_to_string(i1, lang='zxy')
    # i1 = Image.open('金亮IGF.PNG')
    # c1 = pytesseract.image_to_string(i1, lang='chi_sim')
    # c11 = pytesseract.image_to_boxes(i1)
    # #
    # # i2 = Image.open('金亮T-PRL.PNG')
    # i2 = Image.open('金亮T-PRL2.png')
    # c2 = pytesseract.image_to_string(i2, lang='chi_sim')
    # c21 = pytesseract.image_to_boxes(i2, lang='chi_sim')
    # c22 = pytesseract.image_to_data(i2, lang='chi_sim', output_type='data.frame')
    # i2.crop((10, 26, 22, 38)).show()
    # # pytesseract.image_to_boxes(i2)
    # #
    # # i3 = Image.open('金亮T-PRL1.PNG')
    # # c3 = pytesseract.image_to_string(i3, config='digits')
    # # # pytesseract.image_to_boxes(i2)
    # #
    # # i_path = r'E:\zxy\jl\IGF.PNG'
    # # img = cv2.imread(i_path)
    # # # cv2.imshow('b', i1)
    #
    # i3 = Image.open('金亮GH谷值1.PNG')
    # c3 = pytesseract.image_to_string(i3, lang='chi_sim')
    # c31 = pytesseract.image_to_string(i3, config='digits')

    # p0 = r'E:\zxy\2016.1\\'
    # px = r'E:\zxy\tif\\'
    # for p in os.listdir(p0):
    #     p1 = p0 + p + '\\'
    #     for f in os.listdir(p1):
    #         if f.endswith('.PNG'):
    #             i = Image.open(p1 + f)
    #             i.save(px + p + '_' + f.replace('PNG', 'tif'))
    #             print(p + '_' + f.replace('PNG', 'tif'))


def read_xml():
    # path = r'E:\zxy\200615\肢端肥大症\\'
    path = r'E:\zxy\200621\\'

    def get_all_file(path):
        folder_dic = dict()
        for name in os.listdir(path):
            file_dic = dict()
            path1 = path + name + '\\'
            for file in os.listdir(path1):
                file_dic[file] = path1 + file

            folder_dic[name] = file_dic
        df1 = pd.DataFrame(folder_dic).T
        df2 = df1['首页']
        df21 = df1[pd.isna(df1['首页'])]

        from collections import Counter
        C = Counter()
        for f in folder_dic:
            C.update(list(folder_dic[f].keys()))
            for i in folder_dic[f]:
                if i.startswith('首页') and i != '首页':
                    print(f, i)

        C = Counter()
        for f in folder_dic:
            if '手术.doc' not in folder_dic[f]:
                print(f)
            C.update([x for x in list(folder_dic[f].keys()) if x.startswith('手术')])
            for i in folder_dic[f]:
                if i.startswith('手术') and i != '手术.doc':
                    print(f, i)
        return folder_dic

    fd = get_all_file(path)

    def get_xml(fd):
        def read_xml_no_file_name(xml1):
            x = xmlDoc.parse(xml1)
            x1 = x.find('StructuredBody/NInstanceData/ScatterData/Component/Section/Composite/SubItems')
            # x2 = x1.findall('MIDictionary/Code[@DisplayName="手术名称1"]')
            xml_data = dict()
            for child in x1:
                display_name = child.find('Code').get('DisplayName')
                if display_name.startswith('手术日期') or display_name in ['姓名', '年龄', '出生日期', '病案号']:
                    xml_data[display_name] = child.find('Value').text
                if display_name in ['ABO血型', '性别', 'Rh血型'] or display_name.startswith('手术名称'):
                    choice = child.find('Value/Choice')
                    xml_data[display_name] = choice.get('DisplayName') if choice is not None else None

            return xml_data

        def read_xml_xmlfile(xml1):
            x = xmlDoc.parse(xml1)
            position_dic = {
                '出生日期': (5, 4),
                '姓名': (5, 2),
                '年龄': (5, 6),
                '手术名称1': (43, 1),
                '手术日期1': (43, 0),
                '病案号': (4, 5)
            }

            # x2 = x1.findall('MIDictionary/Code[@DisplayName="手术名称1"]')
            xml_data = dict()
            for k, v in position_dic.items():
                xml_data[k] = x.getroot()[v[0]][v[1]].text

            return xml_data

        def get_xml_data(name, fd):
            d1 = fd[name]
            for f in d1:
                if f == '首页':
                    return read_xml_no_file_name(d1[f])
                if f == '首页.xml':
                    return read_xml_xmlfile(d1[f])
            return dict()

        xml_dic = dict()
        patient_count = len(fd)
        i = 0
        for name in fd:
            i += 1
            if i % 10 == 0:
                print('%s / %s, %s' % (i, patient_count, name))
            xml_dic[name] = get_xml_data(name, fd)
        df1 = pd.DataFrame(xml_dic).T
        df1['年龄'] = df1['年龄'].str.split('岁', expand=True)[0]
        for date_name in ['出生日期', '手术日期1', '手术日期2', '手术日期3', '手术日期4', '手术日期5']:
            df1[date_name] = df1[date_name].astype(str).map(lambda x: x[:10] if x else None)

        return df1, xml_dic

    def xml_output():
        xml_data = get_xml(fd)
        xml_data[0].to_excel('xml_数据_200625.xlsx')
        import xlwings as xw
        wb = xw.Book('肢大临床录入数据.xlsx')
        st = wb.sheets[1]
        dfx = xml_data[0]
        for i in range(3, 308):
            id = st.range(i, 2).value
            if isinstance(id, float):
                id = str(int(id))

            print(i, id)
            try:
                xdic = dfx[dfx['病案号'] == id].iloc[0, :].to_dict()
            except IndexError:
                print(id, 'error')
                continue
            st.range(i, 6).value = xdic['年龄']
            st.range(i, 7).value = xdic['出生日期']
            st.range(i, 11).value = xdic['手术日期1']

    def get_coordinate2(i1):
        d1 = pytesseract.image_to_data(i1, lang='chi_sim', output_type=Output.DATAFRAME)
        d2 = d1[d1.text == '项'].iloc[0, :]
        loc_left = d2.loc['left']
        loc_top = d2.loc['top']
        loc_bottom = d2.loc['top'] + d2.loc['height']
        rows = [(loc_top - 5, loc_bottom + 5)]
        row_border = loc_bottom + 9
        while row_border + 18 < i1.size[1]:
            rows.append((row_border + 1, row_border + 23))
            row_border += 39
        columns = [(loc_left - 2, loc_left + 115 + 2)]
        left_border = loc_left + 178

        def get_year_pos(i_times):
            boxes = pytesseract.image_to_boxes(i_times.convert('L'), lang='chi_sim').split('\n')
            s0 = ''
            loc_2_left = -1
            for b in boxes:
                # print(b)
                s0 += b.split()[0]
                if len(s0) == 1:
                    loc_2_left = b.split()[1]
                    if s0 != '2':
                        s0 = ''
                        loc_2_left = -1
                        continue
                if len(s0) == 2:
                    if s0 != '20':
                        s0 = ''
                        loc_2_left = -1
                        continue
                if len(s0) == 3:
                    if s0[2] not in ['0', '1']:
                        s0 = ''
                        loc_2_left = -1
                        continue
                    return int(loc_2_left)
            return -1

        year_pos = -1
        left_temp = left_border - 10
        error = 0
        while year_pos < 0:
            # print('error', error)
            error += 1
            left_temp += 10
            if error > 5:
                raise IndexError('Time Columns Error')
            i_times = i1.crop((left_temp, rows[0][0], left_temp + 125, rows[0][1]))
            year_pos = get_year_pos(i_times)
        left_border = left_temp + year_pos - 8
        while left_border + 113 < i1.size[0]:
            columns.append((left_border + 1, left_border + 113))
            left_border += 125

        return rows, columns

    def test_show(ri, ci, i1, rows, columns):
        col = columns[ci]
        row = rows[ri]
        print(pytesseract.image_to_string(i1.crop((col[0], row[0], col[1], row[1])).convert('L'), lang='chi_sim', config='--psm 7 digits_'))
        i1.crop((col[0], row[0], col[1], row[1])).show()

    def get_coordinate(i1):
        d1 = pytesseract.image_to_data(i1, lang='chi_sim', output_type=Output.DATAFRAME)
        d11 = d1[d1.text == '项']
        r0 = 0
        while d11.shape[0] == 0:
            r0 += 30
            if r0 > 120:
                raise IndexError('No 项')
            d1 = pytesseract.image_to_data(i1.crop((0, 0, i1.size[0], r0)), lang='chi_sim',
                                           output_type=Output.DATAFRAME)
            d11 = d1[d1.text == '项']
        d2 = d11.iloc[0, :]
        loc_left = d2.loc['left']
        loc_top = d2.loc['top']
        loc_bottom = d2.loc['top'] + d2.loc['height']
        rows = [(loc_top - 5, loc_bottom + 5)]
        boxes = pytesseract.image_to_boxes(i1.crop((0, rows[0][0], i1.size[0], rows[0][1])), lang='chi_sim', config='--psm 7').split(
            '\n')

        def get_loc_qu(boxes):
            for b in boxes:
                if b.split()[0] in ['曲', '线', '图']:
                     return int(b.split()[1])
            raise IndexError('曲 not found')

        try:
            loc_qu = get_loc_qu(boxes)
        except IndexError:
            boxes = pytesseract.image_to_boxes(i1.crop((80, rows[0][0], i1.size[0], rows[0][1])), lang='chi_sim',
                                               config='--psm 7').split(
                '\n')
            loc_qu = get_loc_qu(boxes)

        def get_year_columns():
            def get_columns(boxes):
                s0 = ''
                loc_2_left = -1
                for b in boxes:
                    # print(b)
                    s0 += b.split()[0]
                    if len(s0) == 1:
                        loc_2_left = b.split()[1]
                        if s0 != '2':
                            s0 = ''
                            loc_2_left = -1
                            continue
                    if len(s0) == 2:
                        if s0 != '20':
                            s0 = ''
                            loc_2_left = -1
                            continue
                    if len(s0) == 3:
                        if s0[2] in ['0', '1', '2']:
                            return int(loc_2_left)
                        else:
                            s0 = ''
                            loc_2_left = -1
                return

            # loc_2 = get_columns(boxes1) + loc_qu + 45
            # loc_2_list = [loc_2]
            # while loc_2 + 110 < i1.size[0]:
            #     print(loc_2)
            #     boxes1 = pytesseract.image_to_boxes(i1.crop((loc_2 + 110, rows[0][0], i1.size[0], rows[0][1])),
            #                                         lang='chi_sim', config='--psm 7').split(
            #         '\n')
            #     try:
            #         loc_2 += get_columns(boxes1) + 110
            #         loc_2_list.append(loc_2)
            #     except IndexError:
            #         break
            #     except TypeError:
            #         break
            boxes1 = pytesseract.image_to_boxes(i1.crop((loc_qu + 45, rows[0][0], i1.size[0], rows[0][1])),
                                                lang='chi_sim', config='--psm 7').split(
                '\n')

            loc_2 = get_columns(boxes1) + loc_qu + 45
            y1 = pytesseract.image_to_string(i1.crop((loc_2, rows[0][0], loc_2 + 110, rows[0][1])),
                                                lang='chi_sim', config='--psm 7')
            if y1[:2] != '20' or y1[2] not in ('0', '1', '2') or y1[4] != '-':
                boxes1 = pytesseract.image_to_boxes(i1.crop((loc_qu + 60, rows[0][0], i1.size[0], rows[0][1])),
                                                    lang='chi_sim', config='--psm 7').split(
                    '\n')
                loc_2 = get_columns(boxes1) + loc_qu + 60

            loc_2_list = [loc_2]
            if i1.size[0] - loc_2_list[-1] - 115 < 100:
                columns = [(loc_left - 5, loc_qu - 5), (loc_2_list[-1] - 5, i1.size[0])]
                return columns
            boxes2 = pytesseract.image_to_boxes(i1.crop((loc_2 + 115, rows[0][0], i1.size[0], rows[0][1])),
                                                lang='chi_sim', config='--psm 7').split('\n')
            i2 = get_columns(boxes2)
            if len(boxes2) > 5 and not i2:
                boxes2 = pytesseract.image_to_boxes(i1.crop((loc_2 + 110, rows[0][0], i1.size[0], rows[0][1])),
                                                    lang='chi_sim', config='--psm 7').split('\n')
                i2 = get_columns(boxes2)
                interval_2 = i2 + 110
            else:
                interval_2 = i2 + 115
            if interval_2 > 150:
                boxes2 = pytesseract.image_to_boxes(i1.crop((loc_2 + 110, rows[0][0], i1.size[0], rows[0][1])),
                                                    lang='chi_sim', config='--psm 7').split('\n')
                i2 = get_columns(boxes2)
                interval_2 = i2 + 110

            while loc_2 + interval_2 < i1.size[0]:
                loc_2 += interval_2
                # print(loc_2)
                loc_2_list.append(loc_2)
            columns = [(loc_left - 5, loc_qu - 5)] + \
                      [(loc_2_list[i] - 5, loc_2_list[i + 1] - 5) for i in range(len(loc_2_list) - 1)]
            if i1.size[0] - loc_2_list[-1] > 100:
                columns += [(loc_2_list[-1] - 5, i1.size[0])]
            return columns

        columns = get_year_columns()
        # boxr1 = pytesseract.image_to_boxes(i1.crop((columns[0][0], rows[0][1], columns[0][1], rows[0][1] + 30)), lang='chi_sim', config='--psm 7').split('\n')
        # t0 = 10
        # for b in boxr1:
        #     height = int(b.split(' ')[4]) - int(b.split(' ')[2])
        #     if height in range(11, 13):
        #         t0 = int(b.split(' ')[2])
        #         break
        # t1 = t0 + rows[0][1]
        row_cnt = int(round((i1.size[1] - rows[0][1]) / 38, 0))
        avg_row = (i1.size[1] - rows[0][1]) / row_cnt
        for i in range(row_cnt):
            rows += [(rows[0][1] + int(avg_row) * i, rows[0][1] + int(avg_row) * (i + 1))]
        # if t1 + 45 > i1.size[1]:
        #     rows += [(t1 - 5, t1 + 17)]
        # else:
        #     ts = [t1]
        #     while t1 + 45 < i1.size[1]:
        #         t1 += 35
        #     #     boxr2 = pytesseract.image_to_boxes(i1.crop((columns[0][0], t1 + 30, columns[0][1], t1 + 53)),
        #     #                                    lang='chi_sim', config='--psm 7').split('\n')
        #     #     tx = 7
        #     #     for b in boxr2:
        #     #         height = int(b.split(' ')[4]) - int(b.split(' ')[2])
        #     #         if height in range(11, 13):
        #     #             tx = int(b.split(' ')[2])
        #     #             print(b)
        #     #             break
        #     #     t1 += tx + 30
        #         ts.append(t1)
        #     rows += [(ts[i] - 5, ts[i] + 17) for i in range(len(ts))]

        print(rows, columns)

        return rows, columns

    def crop_scale(img, coord, scale):
        w = (coord[2] - coord[0]) * scale
        h = (coord[3] - coord[1]) * scale
        return img.crop((c for c in coord)).resize((w, h))

    @count_time()
    def read_pic(pic_path, scale=2, lang='chi_sim'):
        i1 = Image.open(pic_path)

        rows, columns = get_coordinate(i1)

        data = []
        for ci, col in enumerate(columns):
            # print('Reading Picture %s / %s' % (ci, len(columns)))
            datai = []
            for ri, row in enumerate(rows):
                # img = i1.crop((col[0], row[0], col[1], row[1]))
                img = crop_scale(i1, (col[0], row[0], col[1], row[1]), scale)
                if ci == 0:
                    di = pytesseract.image_to_string(img.convert('L'), lang=lang, config='--psm 7')
                else:
                    if ri == 0:
                        di = pytesseract.image_to_string(img, lang='chi_sim', config='--psm 7').replace('(', '').strip()
                        if len(di) != 16:
                            di = pytesseract.image_to_string(img.convert('L'), lang='chi_sim', config='--psm 7').replace('(', '').strip()
                        if len(di) != 16:
                            di = pytesseract.image_to_string(img.convert('L'), lang='chi_sim',
                                                             config='--psm 7 digits_Y')
                    else:
                        if ri == 2 and 'GI' in pic_path:
                            di = pytesseract.image_to_string(img.convert('L'), lang='chi_sim1',
                                                             config='--psm 7 digits_')
                            if len(di) <= 1:
                                di = pytesseract.image_to_string(img.convert('L'), lang='chi_sim',
                                                                 config='--psm 10 digits_')
                            # print(ri, ci, di)
                        else:
                            di = pytesseract.image_to_string(img.convert('L'), lang='chi_sim',
                                                             config='--psm 10 digits_')
                            if not di or di.startswith('.') or '.' not in di or di.split('.')[-1] == '':
                                di = pytesseract.image_to_string(img.convert('L'), lang='chi_sim1',
                                                                 config='--psm 7 digits_')
                datai.append(di)
            data.append(datai)
        df = pd.DataFrame(data[1:], columns=data[0]).T
        return df

    # p1 = fd['白立新']['ACTH.PNG']
    # i1 = Image.open(p1)
    # t1 = pytesseract.image_to_boxes(i1, lang='chi_sim')
    # d1 = pytesseract.image_to_data(i1, lang='chi_sim', output_type=Output.DATAFRAME)
    # d2 = d1[d1.text == '项'].iloc[0, :]
    # loc_left = d2.loc['left']
    # loc_top = d2.loc['top']
    # loc_bottom = d2.loc['top'] + d2.loc['height']
    #
    # # loc1 = [int(x) for x in t1[t1.find('项'):].split('\n')[0].split(' ')[1:5]]
    # # i_times = i1.crop((loc_left + 178, loc_top - 5, i1.size[0], loc_bottom + 5))
    # i_times = i1.crop((loc_left + 178, loc_top - 5, loc_left + 303, loc_bottom + 5))
    # b1 = pytesseract.image_to_string(i_times, lang='eng')
    @count_time()
    def read_data(name, folder_dic):
        print('-' * 20)
        folder = folder_dic[name]
        pictures = []
        for file in folder:
            if file[-3:] in ['jpg', 'JPG', 'PNG', 'png']:
                pictures.append(file)

        data = dict()
        for pi, p1 in enumerate(pictures):
            print('Reading %s %s / %s' % (p1, pi + 1, len(pictures)))
            pic_path = folder[p1]
            data[p1] = read_pic(pic_path)

        print('-' * 20)
        return data

    def get_pic(folder_dic, first=0, last=100):
        folder_list = sorted(list(folder_dic.keys()), key=pypinyin.lazy_pinyin)
        all_df = []
        errors = []
        t0 = time.perf_counter()
        for i, name in enumerate(folder_list[first:last]):
            try:
                start = time.perf_counter()
                print('Reading %s %s/%s' % (name, i + first + 1, last - first))
                data = read_data(name, folder_dic)
                dfs = []
                for d in data:
                    df = data[d].T
                    df = df.set_index(df.columns[0]).stack(0).reset_index()
                    df.columns = ['date', 'key', 'value']
                    df['name'] = name
                    df['pic'] = d
                    dfs.append(df[['name', 'pic', 'date', 'key', 'value']])
                df = pd.concat(dfs).reset_index(drop=True)
                df['count_time'] = time.perf_counter() - start
            except Exception as e:
                errors.append([name, e, e.args])
                print('*' * 20)
                print(name, e, e.args)
                print('*' * 20)
                continue
            all_df.append(df)
        df = pd.concat(all_df).reset_index(drop=True)
        df.to_excel('图片数据_200621文件夹_%s-%s_%s.xlsx' % (first + 1, last, time.strftime('%y%m%d%H%M%S')), index=False)
        tx = time.perf_counter() - t0
        print('Total Time: %s, Total Success: %s' % (tx, last - first - len(errors)))
        return df, errors, tx

    def get_pic2(folder_dic, folder_list):
        all_df = []
        errors = []
        t0 = time.perf_counter()
        for i, name in enumerate(folder_list):
            try:
                start = time.perf_counter()
                print('Reading %s %s/%s' % (name, i + 1, len(folder_list)))
                data = read_data(name, folder_dic)
                dfs = []
                for d in data:
                    df = data[d].T
                    df = df.set_index(df.columns[0]).stack(0).reset_index()
                    df.columns = ['date', 'key', 'value']
                    df['name'] = name
                    df['pic'] = d
                    dfs.append(df[['name', 'pic', 'date', 'key', 'value']])
                df = pd.concat(dfs).reset_index(drop=True)
                df['count_time'] = time.perf_counter() - start
            except Exception as e:
                errors.append([name, e, e.args])
                print('*' * 20)
                print(name, e, e.args)
                print('*' * 20)
                continue
            all_df.append(df)
        df = pd.concat(all_df).reset_index(drop=True)
        df.to_excel('图片数据_200621文件夹_errorfix_%s.xlsx' % time.strftime('%y%m%d%H%M%S'), index=False)
        tx = time.perf_counter() - t0
        print('Total Time: %s, Total Success: %s' % (tx, len(folder_list) - len(errors)))
        return df, errors, tx

    def test():
        # data_1 = read_data('白立新', fd)
        # data_1 = read_data('白立新', fd)
        # data_2 = read_data('东方云', fd)
        # data_3 = read_data('高梁清', fd)
        # data_4 = read_data('高剑霞', fd)
        data_1 = read_data('郝建', fd)
        # i1 = Image.open(fd['陈坤']['X.PNG'])
        # i1 = Image.open(fd['白立新']['ACTH.PNG'])
        # i1 = Image.open(fd['陈晓华']['X.JPG'])
        # i1 = Image.open(fd['张培玲']['GG.JPG'])
        # i1 = Image.open(fd['冯海侠']['F.JPG'])
        # i1 = Image.open(fd['郝建']['GG.JPG'])
        # i1 = Image.open(fd['胡亚利']['GG.JPG'])
        # i1 = Image.open(fd['黄安东']['X.JPG'])
        i1 = Image.open(fd['李畅']['GI.JPG'])
        i1 = Image.open(fd['王姝']['GG2.PNG'])
        i1 = Image.open(fd['肖俊峰']['X2.JPG'])
        i1 = Image.open(fd['肖俊峰']['A.JPG'])
        i1 = Image.open(fd['许建芹']['GG.JPG'])
        i1 = Image.open(fd['余争先']['GI.JPG'])
        i1 = Image.open(fd['余争先']['X.JPG'])
        i1 = Image.open(fd['卞敬华']['GG.JPG'])
        rows, columns = get_coordinate(i1)
        dfx = read_pic(fd['卞敬华']['GG.JPG'])
        dfx = read_pic(fd['高润娥']['X.JPG'])
        test_show(0, 2, i1, rows, columns)

    dfn1, error1, tx1 = get_pic(fd, 0, 10)
    dfn2, error2, tx2 = get_pic(fd, 0, 305)
    dfe = pd.DataFrame(error2)
    dfe.to_excel('ERROR2.xlsx')
    dfn2['date_len'] = dfn2['date'].map(len)
    dfn21 = dfn2[dfn2.date_len != 16]

    e1 = dfe[0].tolist()
    e2 = dfn21['name'].drop_duplicates().tolist()
    e_list = list(set(e1 + e2))

    dfn3, error3, tx3 = get_pic2(fd, e_list)


