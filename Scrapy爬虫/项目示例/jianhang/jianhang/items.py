# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class JianhangItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    fxyh = scrapy.Field() #发行银行
    cpmc = scrapy.Field() #产品名称
    gmbz = scrapy.Field() #购买币种
    cplx = scrapy.Field() #产品类型
    sylx = scrapy.Field() #收益类型
    tzfw = scrapy.Field() #投资范围
    fxdj = scrapy.Field() #风险等级
    tzqx = scrapy.Field() #投资期限
    yqsy = scrapy.Field() #预期收益
    xsqs = scrapy.Field() #销售起始日期
    xsjz = scrapy.Field() #销售截止日期
    qsje = scrapy.Field() #起始金额
    kfsh = scrapy.Field() #可否赎回
    fxdq = scrapy.Field() #发行地区
