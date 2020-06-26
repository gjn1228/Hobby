# -*- coding: utf-8 -*-
# @Author  : GJN
import pymongo
from mitmproxy import ctx
import json


class Mongodb(object):
    def __init__(self, course_name):
        client = pymongo.MongoClient(host='localhost', port=27017)
        db = client['admin']
        self.collection = db['dxy_test']
        self.data = dict()
        self.course_name = course_name
        self.reset()
        self.count = 0

    def reset(self):
        self.data = dict()
        self.data['course_name'] = self.course_name

    def update_db(self):
        ctx.log.info("-" * 20)
        self.collection.insert_one(self.data)
        ctx.log.info("*" * 20)
        ctx.log.info("Update %s" % self.data.get('c'))
        ctx.log.info("*" * 20)
        # self.data = dict()
        self.reset()
        self.count += 1


    # def update_db(self):
    #     ctx.log.info("-" * 20)
    #     ctx.log.info(str(self.data))
    #     if self.data.get('courseHourName'):
    #         if self.data.get('ts'):
    #             if self.collection.find_one({'courseHourName': self.data.get('courseHourName')}):
    #                 self.collection.update_one({'courseHourName': self.data.get('courseHourName')}, {'$set': self.data})
    #             else:
    #                 self.collection.insert_one(self.data)
    #             ctx.log.info("*" * 20)
    #             ctx.log.info("Update %s" % self.data.get('courseHourName'))
    #             ctx.log.info("*" * 20)
    #             self.data = dict()


M = Mongodb(input('Course Name:'))


def response(flow):
    global M
    ctx.log.info("-----------")
    # d_url = 'https://class.dxy.cn/pcweb/play-record/detail?'
    #
    # if flow.request.url.startswith(d_url):
    #     text = flow.response.text
    #     M.data.update(json.loads(text).get('data'))
    #     M.update_db()

    v_url_1 = 'https://1252348479.vod2.myqcloud.com'
    v_url_2 = 'voddrm.token'
    if flow.request.url.startswith(v_url_1) and v_url_2 in flow.request.url:
        M.data.update({'ts': flow.response.text, 'm_url': flow.request.url, 'c': M.count})
        M.update_db()

















