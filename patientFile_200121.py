# -*- coding: utf-8 -*-
# @Author  : GJN
import xlwings as xw
import pandas as pd
import datetime as dt
import numpy as np

path = r'C:\Users\86135\Downloads\\'
wb1 = xw.Book('TSH瘤临床录入表格(1).xlsx')
wb2 = xw.Book('20200121-录病例.xlsx')

st = wb2.sheets[18]


def get_data_from_st(st):
    id = st.range(1, 2).value
    r = 1

    while st.range(r, 1).value != '促甲状腺激素':
        r += 1
        if r > 30:
            print(st.name)
            raise IndexError

    row_name = []
    while st.range(r, 1).value != '项目范围':
        row_name.append(st.range(r, 1).value)
        r += 1
        if r > 30:
            print(st.name)
            raise IndexError

    r += 1
    dates = []

    while st.range(r, 1).value:
        dates.append(st.range(r, 1).value)
        r += 1
        if r > 100:
            print(st.name)
            raise IndexError

    r += 1

    datas = [[] for x in range(len(row_name))]
    for i in range(len(row_name)):
        for j in range(len(dates)):
            datas[i].append(st.range(r + i, j + 2).value)

    df1 = pd.DataFrame({x: y for x, y in zip(row_name, datas)})
    df1['date'] = dates

    return id, df1


datas = []
for i in range(2, len(wb2.sheets)):
    st = wb2.sheets[i]
    datas.append(get_data_from_st(st))

st1 = wb1.sheets[0]
id_dic = dict()
for r in range(10, 42):
    id_dic[st1.range(r, 1).value] = st1.range(r, 4).value


def get_value(df1, date1):
    df1['dx'] = (df1.date - date1).dt.days
    mindx = df1[df1.dx <= 0]['dx'].max()
    if mindx > 7:
        r_dx = None
        print(i, id, 'No Date dx')
    elif mindx == 0:
        print(i, id, 'Same Day')
        if df1[df1.dx == mindx].iloc[-1, :]['date'].hour < 9:
            r_dx = df1[df1.dx == mindx].index.tolist()[-1]
        else:
            mindx = df1[df1.dx < 0]['dx'].max()
            if mindx > 7:
                r_dx = None
                print(i, id, 'No Date dx')
            elif df1[df1.dx == mindx].shape[0] > 1:
                print(i, id, 'More Than 1 dx')
                r_dx = None
            else:
                r_dx = df1[df1.dx == mindx].index.tolist()[0]
    elif df1[df1.dx == mindx].shape[0] > 1:
        print(i, id, 'More Than 1 dx')
        r_dx = None
    else:
        r_dx = df1[df1.dx == mindx].index.tolist()[0]

    df1['dx3'] = df1.dx - 3
    mindx3 = abs(df1['dx3']).min()
    if df1[abs(df1['dx3']) == mindx3].shape[0] > 1:
        min_dx3_tsh = df1[abs(df1['dx3']) == mindx3][xdic['TSH']].min()
        r_dx3 = df1[abs(df1['dx3']) == mindx3][abs(df1[xdic['TSH']]) == min_dx3_tsh].index.tolist()[0]
        print(i, id, 'More Than 1 dx3')
        # r_dx3 = None
    else:
        r_dx3 = df1[abs(df1['dx3']) == mindx3].index.tolist()[0]

    df1['dx90'] = df1.dx - 90
    mindx90 = abs(df1['dx90']).min()
    if mindx90 > 30:
        print(i, id, 'No Data dx90')
        r_dx90 = None
    elif df1[abs(df1['dx90']) == mindx90].shape[0] > 1:
        print(i, id, 'More Than 1 dx90')
        r_dx90 = None
    else:
        r_dx90 = df1[abs(df1.dx90) == mindx90].index.tolist()[0]

    min_dic = dict()
    for xc in xdic:
        x = xdic[xc]
        min_x = df1[df1.dx > 0][x].min()
        # 选靠近的
        r_x = df1[df1.dx > 0][df1[x] == min_x].index.tolist()[-1]
        if df1[df1.dx > 0][df1[x] == min_x].shape[0] > 1:
            print(i, id, xc, 'More Than 1 guzhi')
        days = df1.loc[r_x, 'dx']
        if days <= 30:
            dx = str(days) + 'd'
        elif days < 365:
            dx = str(round(days / 30, 1)) + 'm'
        else:
            dx = str(round(days / 365, 1)) + 'y'
        min_dic[xc] = [min_x, dx]

    return r_dx, r_dx3, r_dx90, min_dic


##
wb3 = xw.Book('TSH瘤临床录入表格_1.xlsx')
st3 = wb3.sheets[0]
xdic = {'TSH': '促甲状腺激素',
 'FT4': '游离甲状腺素',
 'T3': '三碘甲状腺原氨酸',
 'T4': '甲状腺素',
 'FT3': '游离三碘甲状腺原氨酸'}

col_dic = {'before': {'TSH': 39,
 'FT4': 40,
 'T3': 41,
 'T4': 42,
 'FT3': 43},
           '3d': {'TSH': 44,
 'FT4': 45,
 'T3': 46,
 'T4': 47,
 'FT3': 48},
           '3m': {'TSH': 49,
                  'FT4': 50,
                  'T3': 51,
                  'T4': 52,
                  'FT3': 53},
'min': {'TSH': 54,
                  'FT4': 58,
                  'T3': 60,
                  'T4': 62,
                  'FT3': 56}
           }
row_dic = dict()
for r in range(10, 42):
    row_dic[st3.range(r, 1).value] = r

i = 0


def go(i):
    id = datas[i][0]
    date1 = dt.datetime.strptime(id_dic[id], '%Y.%m.%d')
    df1 = datas[i][1]
    r_dx, r_dx3, r_dx90, min_dic = get_value(df1, date1)

    row = row_dic[id]
    ty_l = ['before', '3d', '3m']
    for t in range(3):
        col_name = ty_l[t]
        data_r = [r_dx, r_dx3, r_dx90][t]

        for x in col_dic[col_name]:
            if not data_r:
                v = 'n'
            else:
                v = df1.loc[data_r, xdic[x]]
                if np.isnan(v):
                    v = 'n'
            st3.range(row, col_dic[col_name][x]).value = v

    for x in col_dic['min']:
        v, d = min_dic[x]
        if np.isnan(v):
            v = 'n'
        st3.range(row, col_dic['min'][x]).value = v
        st3.range(row, col_dic['min'][x] + 1).value = d

    return df1


dfs = []
for i in range(len(datas)):
    df = go(i)
    dfs.append(df)


