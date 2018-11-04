# -*- coding: utf-8 -*-
import scrapy
from scrapy.selector import HtmlXPathSelector
import re
import os
import sys
import json
from scrapy.http import Request
from scrapy.spiders import Spider
from googlesearch.items import GooglesearchItem
from scrapy import signals
from scrapy.xlib.pydispatch import dispatcher
from argparse import ArgumentParser


class GooglesearchspiderSpider(scrapy.Spider):
	
	# 識別名，在terminal中填寫scrapy crawl googlesearchspider
	name = 'googlesearchspider'
	allowed_domains = []

	# 填寫要訪問的url
	start_urls = ['https://www.google.com.tw/search?q=vidiu&ei=YgnfW_7INoS18QW1z5mYCg&start=0&sa=N&ved=0ahUKEwj-geWngLveAhWEWrwKHbVnBqM4ChDy0wMIgwE&biw=1280&bih=698']
	
	def __init__(self):
		self.frequency_count = dict(VidiU = 0,Beam = 0,LiveShell = 0,Tricaster = 0,Livestream = 0,)
		self.base_url = 'https://www.google.com.tw'
		dispatcher.connect(self.spider_closed, signals.spider_closed)

	def frequency_renew(self,key_word,doc):
		self.frequency_count[key_word]+=doc.count(key_word)

	def parse(self, response):

		# 實例化儲存物件
		item = GooglesearchItem()

		hxs = HtmlXPathSelector(response)
		
		
		# 抓取各搜尋資料之資訊
		for data in hxs.select('//div[@id="ires"]//div[@class="g"]//div[@class="s"]//span[@class="st"]'):
			item['description'] = ''.join(data.select('text()').extract())
			
			list(map(lambda key_word:self.frequency_renew(key_word,item['description']), self.frequency_count))
			yield item


		# 抓取下一頁之url
		next_page = hxs.select('//table[@id="nav"]//td[contains(@class, "b") and position() = last()]/a')

		# 訪問下一頁之url
		if next_page:
			request_url = self.base_url+next_page.select('.//@href').extract()[0]
			yield Request(url=request_url, callback=self.parse)

	def spider_closed(self, spider):
		print("END")
		print('frequency_count:',self.frequency_count)
		



