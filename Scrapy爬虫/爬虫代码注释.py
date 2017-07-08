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

#以上代码需要复制

class AzhaoshangSpider(scrapy.Spider):
    name = "azhaoshang" #爬虫名称，后面运行爬虫的时候需要输入此名称
    allowed_domains = ["cmbchina.com"]  #网站域名
    #以上3行代码不需要复制
    start_urls = (
        'http://www.cmbchina.com/cfweb/Personal/Default.aspx',
    )                                 #产品列表页面的网址。请修改为自己爬取网站的产品列表页面URL
    '''
    情形一：产品列表页是静态页面（比较少见），也就是火狐浏览器里右键查看网页源文件，能直接看到网页上显示的内容;
    情形二：产品列表页是动态页面，火狐浏览器里右键查看网页源文件，里面查看不到网页上显示的内容，网页内容是经过javascript动态生成的;

    对于情形一，不需要下面__init__和to_page_bottom两个函数，因此不要复制
    对于情形二，需要调用__init__函数，因此要把__init__函数复制过去；而to_page_bottom需要视情况采用。
    '''

    def __init__(self, *args, **kwargs):
        '''
        以下代码是情况二样例代码
        '''
        #super中的第一个参数：AzhaoshangSpider，需要修改为自己的名称。即上面的class名称
        super(AzhaoshangSpider, self).__init__(*args, **kwargs)
        # 如果把PhantomJS替换为Firefox,爬虫会启动一个Firefox浏览器，能看到爬虫运行时浏览器的实时变化
        self.browser = webdriver.PhantomJS()

        self.browser.get(self.start_urls[0])
        #self.browser2.refresh() #刷新网页，个别网站（建行）需要刷新一次才能显示完整内容
        self.browser.implicitly_wait(10) #等待超时时间
        time.sleep(5)
        self.to_page_bottom()
        self.prod_urls = list()  #用来保存所有产品详情页的网址


    def to_page_bottom(self):  # 该函数的作用是滚动到网页底部，因为招行网站需要滚动到底部来加载更多产品信息，视情况采用。
        for _ in range(6):
            self.browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)

    def parse(self, response):

        '''
        函数作用：解析动态产品列表页
        重点:
        （1）self.browser.find_elements_by_xpath('//div/span[@class="inlineBriefName"]/a')，
        获得列表页中所有产品详情页的网址(href属性)所在的elements（通常是a元素）的列表，需要根据爬取的网站具体情况修改。
        其中xpath函数内是网页中需要获取的元素，获取方式见：《附录（一）》
        （2）next_page = self.browser.find_element_by_xpath('//div[@id="pager"]/div/a[contains(text(), "下一页")]')，
        定位翻页按钮，需要根据爬取的网站具体情况修改。
        '''
        self.log('开始运行爬虫')
        for a in self.browser.find_elements_by_xpath('//div/span[@class="inlineBriefName"]/a'):     #需要修改
            prod_url = a.get_attribute('href')
            self.prod_urls.append(prod_url)

        #以下翻页代码针对动态产品列表网页，如果是静态网页，需要重写！
        self.log('----------------翻页------------------')
        next_page = self.browser.find_element_by_xpath('//div[@id="pager"]/div/a[contains(text(), "下一页")]') #需要修改
        next_page.click()
        time.sleep(5)
        self.to_page_bottom()
        body = self.browser.page_source
        yield scrapy.Request(url=self.browser.current_url, body=body, callback=self.parse)

        for u in self.prod_urls:
            yield scrapy.Request(u,callback=self.prod_parse)



    '''
    prod_parse函数用来解析理财产品详情页
    下面有两个prod_parse函数，其中第一个用来演示详情页是静态网页的情形；
    第二个演示详情页是动态网页的情形；
    根据实际情况复制一个就行。
    '''
    #（一）静态网页
    def prod_parse(self, response):
        '''
        解析产品详情页（http://www.cmbchina.com/cfweb/personal/productdetail.aspx?code=1200052）的函数;
        ZhaoshangItem已经预先在items.py中定义，其中fxyh,cpmc,... 每个字段都有特定含义，
        如果网页上有与这里某个含义对应的字段就需要提取，没有含义对应的字段就用''表示缺失,而这里没有列出的字段都不需要提取；

        xpath函数内的内容都需要修改
        例如下面的语句：
        product['cpmc'] = ''.join(response.xpath('//div[@style="padding: 0px 3px;"]/text()').extract()).strip().encode('mbcs')
        其中'cpmc'表示产品名, xpath('//div[@style="padding: 0px 3px;"]/text()')语句可以提取出产品名称,
        .extract()).strip().encode('mbcs')不需要修改,encode('mbcs')的作用是处理中文乱码。
        '''

        sys.path.append('c:\\zhaoshang')
        from zhaoshang.items import ZhaoshangItem
        product = ZhaoshangItem()

        product['fxyh'] = '招商银行'.encode('mbcs') #爬取的银行名称，需要修改
        product['cpmc'] = ''.join(response.xpath('//div[@style="padding: 0px 3px;"]/text()').extract()).strip().encode('mbcs')
        product['gmbz'] = ''.join(response.xpath('//li[9]/span/text()').extract()).strip().encode('mbcs')
        product['sylx'] = ''.join(response.xpath('//li[6]/span/text()').extract()).strip().encode('mbcs')
        product['tzfw'] = ''  #详情页上没有对应字段，赋值为空字符串，表示缺失值
        product['fxdj'] = ''.join(response.xpath('//li[7]/span/text()').extract()).strip().encode('mbcs')
        product['tzqx'] = ''.join(response.xpath('//div[@id="ctl00_content_panel"]//tr[4]//p[@style="line-height:125%;"]/span[1]/text()').extract()).strip().encode('mbcs')
        product['yqsy'] = ''.join(response.xpath('//li[5]/span/text()').extract()).strip().encode('mbcs')
        product['xsqs'] = ''.join(response.xpath('//li[2]/span/text()').extract()).strip().encode('mbcs')
        product['xsjz'] = ''.join(response.xpath('//li[3]/span/text()').extract()).strip().encode('mbcs')
        product['qsje'] = ''
        product['kfsh'] = ''
        product['fxdq'] = ''
        return product

    #（二）动态网页
    def prod_parse(self, response):
        '''
        解析产品详情页（http://finance.ccb.com/cn/finance/ProductDetails.html?PRODUCT_ID=ZHQYBB20160400042）的函数;
        JianhangItem已经预先在items.py中定义，其中fxyh,cpmc,... 每个字段都有特定含义，
        如果网页上有与这里某个含义对应的字段就需要提取，没有含义对应的字段就用''表示缺失,而这里没有列出的字段都不需要提取；

        xpath函数内的内容都需要修改
        例如下面的语句：
        product['cpmc'] = self.browser2.find_element_by_xpath('//table[@class="cell"]//td[@id="name2"]').text.encode('mbcs')
        其中'cpmc'表示产品名, find_element_by_xpath('//table[@class="cell"]//td[@id="name2"]').text 语句可以提取出产品名称,
        .text.strip().encode('mbcs')不需要修改,encode('mbcs')的作用是处理中文乱码。
        '''
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
        product['tzqx'] = self.browser2.find_element_by_xpath('//*[@id="investPeriod2"]').text.estrip().encode('mbcs')
        product['yqsy'] = self.browser2.find_element_by_xpath('//td[@id="yieldRate2"]').text.strip().encode('mbcs')
        product['xsqs'] = self.browser2.find_element_by_xpath('//*[@id="collBgnDate2"]').text.strip().encode('mbcs')
        product['xsjz'] = ''
        product['qsje'] = self.browser2.find_element_by_xpath('//*[@id="purFloorAmt2"]').text.strip().encode('mbcs')
        product['kfsh'] = self.browser2.find_element_by_xpath('//*[@id="proMode"]').text.strip().encode('mbcs')
        product['fxdq'] = self.browser2.find_element_by_xpath('//*[@id="saleCitys"]').text.strip().encode('mbcs')
        self.browser2.close()
        return product






