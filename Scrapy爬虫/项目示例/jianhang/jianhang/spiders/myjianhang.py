# -*- coding: utf-8 -*-
"""
@author: qiulongsheng@beyondsoft.com
"""


import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import scrapy
from selenium import webdriver
import time


class MyjianhangSpider(scrapy.Spider):
    name = "myjianhang"
    allowed_domains = ["ccb.com"]
    start_urls = [
        "http://finance.ccb.com/cn/finance/product.html"
    ]

    def __init__(self, *args, **kwargs):
        super(MyjianhangSpider, self).__init__(*args, **kwargs)
        self.browser = webdriver.Firefox()
        self.browser.get(self.start_urls[0])
        self.browser.refresh()
        self.browser.implicitly_wait(10)
        time.sleep(5)
        self.prod_urls = list()  #用来保存所有产品详情页的网址

    def parse(self, response):

        self.log('开始运行爬虫')
        for td in self.browser.find_elements_by_xpath('//td[@class="list_title"]'):     #需要修改
            a = td.find_element_by_xpath('./a')
            prod_url = a.get_attribute('href')
            self.prod_urls.append(prod_url)

        next_page = self.browser.find_element_by_xpath('//span[@id="next_span"]/a[contains(text(), "下一页")]') #需要修改
        next_page.click()
        time.sleep(5)
        body = self.browser.page_source
        yield scrapy.Request(url=self.browser.current_url, body=body, callback=self.parse)
        self.log('----------------翻页------------------')

        for u in self.prod_urls:
            yield scrapy.Request(u,callback=self.prod_parse)

    def prod_parse(self, response):
        sys.path.append('c:\\jianhang')
        from jianhang.items import JianhangItem
        product = JianhangItem()

        self.browser2 = webdriver.PhantomJS()
        self.browser2.get(response.url)
        time.sleep(5)

        product['fxyh'] = '建设银行'.encode('mbcs')
        product['cpmc'] = self.browser2.find_element_by_xpath('//table[@class="cell"]//td[@id="name2"]').text.strip().encode('mbcs')
        product['gmbz'] = self.browser2.find_element_by_xpath('//*[@id="currencyType"]').text.strip().encode('mbcs')
        product['cplx'] = ''
        product['sylx'] = self.browser2.find_element_by_xpath('//*[@id="yieldSpec"]').text.strip().encode('mbcs')
        product['tzfw'] = ''
        product['fxdj'] = self.browser2.find_element_by_xpath('//*[@id="riskLevel"]').text.strip().encode('mbcs')
        product['tzqx'] = self.browser2.find_element_by_xpath('//*[@id="investPeriod2"]').text.strip().encode('mbcs')
        product['yqsy'] = self.browser2.find_element_by_xpath('//td[@id="yieldRate2"]').text.strip().encode('mbcs')
        product['xsqs'] = self.browser2.find_element_by_xpath('//*[@id="collBgnDate2"]').text.strip().encode('mbcs')
        product['xsjz'] = ''
        product['qsje'] = self.browser2.find_element_by_xpath('//*[@id="purFloorAmt2"]').text.strip().encode('mbcs')
        product['kfsh'] = self.browser2.find_element_by_xpath('//*[@id="proMode"]').text.strip().encode('mbcs')
        product['fxdq'] = self.browser2.find_element_by_xpath('//*[@id="saleCitys"]').text.strip().encode('mbcs')
        self.browser2.close()
        return product
