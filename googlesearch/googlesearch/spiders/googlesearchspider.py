# -*- coding: utf-8 -*-
import scrapy
from scrapy.selector import HtmlXPathSelector
import re
from scrapy.http import Request
from scrapy.spiders import Spider
from googlesearch.items import GooglesearchItem
from scrapy import signals
from scrapy.xlib.pydispatch import dispatcher
from argparse import ArgumentParser
from scrapy.selector import Selector
from urllib.parse import urlparse,parse_qsl
import w3lib
from scrapy.http import FormRequest
import requests
from scrapy.spider import BaseSpider
from scrapy.utils.misc import arg_to_iter




class GooglesearchspiderSpider(scrapy.Spider):
	
	# 識別名，在terminal中填寫scrapy crawl googlesearchspider
	name = 'googlesearchspider'
	allowed_domains = []

	# 填寫要訪問的url
	#start_urls = ['https://www.google.com.tw/search?q=livebox']

	base_url_fmt = 'https://www.google.com.tw/search?q={query}'

	def __init__(self,queries):
		self.queries = queries.split(';')
		self.frequency_count = dict(vidiu = 0,beam = 0,liveshell = 0,tricaster = 0,livestream = 0)
		self.base_url = 'https://www.google.com.tw'
		self.url2weight = {}
		self.done = False
		
		dispatcher.connect(self.spider_closed, signals.spider_closed)

	def start_requests(self):

		for query in arg_to_iter(self.queries):
			url = self.make_google_search_request(query)
			
			yield Request(url=url,callback = self.parse)

	def make_google_search_request(self, query):

		return self.base_url_fmt.format(query='+'.join(query.split()).strip('+'))

	def frequency_renew(self,key_word,doc,score):
		doc = doc.lower()
		self.frequency_count[key_word]+=(doc.count(key_word)*score)

	def parse(self, response):
		
		
		
		hxs = HtmlXPathSelector(response)
		
		for sel in hxs.select('//div[@id="ires"]//div[@class="g"]'):

			
			name = u''.join(sel.select(".//div[@class='s']//text()").extract())
			
			url = sel.select('.//a/@href').extract()[0]
			
			
			
			
			
			if url.startswith('http'):
				formdata = {'name': url}

				
				yield FormRequest(url = 'https://checkpagerank.net/check-page-rank.php',
							formdata=formdata,
							meta={'url':url},
							callback=self.parsePageRank)
				

				
			else:
				url = self.base_url+url
				formdata = {'name': url}

				
				yield FormRequest(url = 'https://checkpagerank.net/check-page-rank.php',
							formdata=formdata,
							meta={'url':url},
							callback=self.parsePageRank)
				

			if self.done:
				self.done = False
				yield Request(url = url,callback = self.parse_page,meta={'name':name})

			
			

		
		# 抓取下一頁之url
		next_page = hxs.select('//table[@id="nav"]//td[contains(@class, "b") and position() = last()]/a')

		# 訪問下一頁之url
		if next_page:
			request_url = self.base_url+next_page.select('.//@href').extract()[0]
			yield Request(url=request_url, callback=self.parse)
		

	def parsePageRank(self,response):
		#response = fromstring(response.content)
		
		
		url = response.meta['url']
		score = response.xpath('//*[@id="pdfdiv"]/div[5]/div/h2/font[2]/b/text()').extract()
		
		

		if type(score)==list:
			score = score[0].replace('/10','')
		else:
			score.replace('/10','')

		if score:
			self.url2weight[url] = int(score)
		else:
			self.url2weight[url] = 1

		self.done = True
		

	def parse_page(self,response):

		item = GooglesearchItem()
		sel = Selector(response)
		name = response.meta['name']
		url = response.url

		
		html = w3lib.html.remove_tags(
			w3lib.html.remove_tags_with_content(
				sel.xpath('//body').extract()[0],
				which_ones=('script',)
			)
		)
		
		pattern = re.compile(r"\s+")
		html = pattern.sub(" ", html)
		item['name'] = name
		item['description'] = html
		item['url'] = url		
		item['score'] = self.url2weight[url]
		list(map(lambda key_word:self.frequency_renew(key_word,item['description'],item['score']), self.frequency_count))
		list(map(lambda key_word:self.frequency_renew(key_word,item['name'],item['score']), self.frequency_count))

		yield item
		
	def spider_closed(self, spider):
		print("END")
		print('得分:',self.frequency_count)

