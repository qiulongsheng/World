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

class AzhaoshangSpider(scrapy.Spider):
    name = "azhaoshang"
    allowed_domains = ["cmbchina.com"]
    start_urls = (
        'http://www.cmbchina.com/cfweb/Personal/Default.aspx',
    )

    def __init__(self, *args, **kwargs):
        super(AzhaoshangSpider, self).__init__(*args, **kwargs)
        self.browser = webdriver.PhantomJS()
        self.browser.get(self.start_urls[0])
        self.browser.implicitly_wait(10)
        time.sleep(3)
        self.to_page_bottom()
        self.prod_urls = list()  #用来保存所有产品详情页的网址


    def to_page_bottom(self):  # 该函数的作用是滚动到网页底部，因为招行网站需要滚动到底部来加载更多产品信息，视情况采用。
        for _ in range(6):
            self.browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)

    def parse(self, response):
        self.log('开始运行爬虫')
        for a in self.browser.find_elements_by_xpath('//div/span[@class="inlineBriefName"]/a'):     #需要修改
            prod_url = a.get_attribute('href')
            self.prod_urls.append(prod_url)
        self.log('----------------翻页------------------')
        next_page = self.browser.find_element_by_xpath('//div[@id="pager"]/div/a[contains(text(), "下一页")]') #需要修改
        next_page.click()
        time.sleep(3)
        self.to_page_bottom()
        body = self.browser.page_source
        yield scrapy.Request(url=self.browser.current_url, body=body, callback=self.parse)


        for u in self.prod_urls:
            yield scrapy.Request(u,callback=self.prod_parse)

    def prod_parse(self, response):
        sys.path.append('C:\\zhaoshang')
        from zhaoshang.items import ZhaoshangItem
        product = ZhaoshangItem()

        product['fxyh'] = '招商银行'.encode('mbcs')
        product['cpmc'] = ''.join(response.xpath('//div[@style="padding: 0px 3px;"]/text()').extract()).strip().encode('mbcs')
        product['gmbz'] = ''.join(response.xpath('//li[9]/span/text()').extract()).strip().encode('mbcs')
        product['sylx'] = ''.join(response.xpath('//li[6]/span/text()').extract()).strip().encode('mbcs')
        product['tzfw'] = ''
        product['fxdj'] = ''.join(response.xpath('//li[7]/span/text()').extract()).strip().encode('mbcs')
        product['tzqx'] = ''.join(response.xpath('//div[@id="ctl00_content_panel"]//tr[4]//p[@style="line-height:125%;"]/span[1]/text()').extract()).strip().encode('mbcs')
        product['yqsy'] = ''.join(response.xpath('//li[5]/span/text()').extract()).strip().encode('mbcs')
        product['xsqs'] = ''.join(response.xpath('//li[2]/span/text()').extract()).strip().encode('mbcs')
        product['xsjz'] = ''.join(response.xpath('//li[3]/span/text()').extract()).strip().encode('mbcs')
        product['qsje'] = ''
        product['kfsh'] = ''
        product['fxdq'] = ''
        return product



