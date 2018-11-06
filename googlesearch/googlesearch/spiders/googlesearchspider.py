# -*- coding: utf-8 -*-

# 爬蟲
import scrapy
from scrapy.selector import HtmlXPathSelector
from scrapy.selector import Selector
from scrapy.http import Request
from scrapy.spiders import Spider
from scrapy import signals
from scrapy.xlib.pydispatch import dispatcher
from scrapy.http import FormRequest
from googlesearch.items import GooglesearchItem
import requests
from scrapy.utils.misc import arg_to_iter

# 處理網頁資料
import re
import w3lib

class GooglesearchspiderSpider(scrapy.Spider):
	
	# 識別名，在terminal中填寫scrapy crawl googlesearchspider
	name = 'googlesearchspider'
	allowed_domains = []
	
	base_url_fmt = 'https://www.google.com.tw/search?q={query}'

	def __init__(self,queries,brandname):
		# 查詢詞列表
		self.queries = queries.split(';')

		# 欲計算之品牌
		self.frequency_count = {brand:0 for brand in brandname.split(';')}
		
		# PageRank權重
		self.url2weight = {}

		# Base URL
		self.base_url = 'https://www.google.com.tw'

		dispatcher.connect(self.spider_closed, signals.spider_closed)

	def start_requests(self):

		# 依序將查詢詞投入Google搜尋並進行解析
		for query in arg_to_iter(self.queries):
			url = self.make_google_search_request(query)
			
			yield Request(url=url,callback = self.parse)

	def make_google_search_request(self, query):

		# 製作Google查詢之URL
		return self.base_url_fmt.format(query='+'.join(query.split()).strip('+'))

	def frequency_renew(self,key_word,doc,score):

		# 計算品牌熱度
		doc = doc.lower()
		self.frequency_count[key_word]+=(doc.count(key_word)*score)

	def parse(self, response):
		
		hxs = HtmlXPathSelector(response)
		
		for sel in hxs.select('//div[@id="ires"]//div[@class="g"]'):

			# 主題與簡介
			name = u''.join(sel.select(".//div[@class='s']//text()").extract())
			
			# 各筆網頁之URL
			url = sel.select('.//a/@href').extract()[0]
			
			
			# 查詢各筆網頁之PageRank
			if url.startswith('http'):
				formdata = {'name': url}
				
				
				while url not in self.url2weight:
					yield FormRequest(url = 'https://checkpagerank.net/check-page-rank.php',
									formdata=formdata,
									meta={'url':url},
									callback=self.parsePageRank)			
			else:
				url = self.base_url+url
				formdata = {'name': url}
				
				while url not in self.url2weight:
			
					yield FormRequest(url = 'https://checkpagerank.net/check-page-rank.php',
								formdata=formdata,
								meta={'url':url},
								callback=self.parsePageRank)
				
			
			# 解析各筆網頁內容
			if url in self.url2weight:	
				yield Request(url = url,callback = self.parse_page,meta={'name':name})


			
			

		
		# 抓取Google搜尋下一頁之url
		next_page = hxs.select('//table[@id="nav"]//td[contains(@class, "b") and position() = last()]/a')
		if next_page:
			request_url = self.base_url+next_page.select('.//@href').extract()[0]
			yield Request(url=request_url, callback=self.parse)
		

	# 查詢PageRank
	def parsePageRank(self,response):
		
		url = response.meta['url']
		score = response.xpath('//*[@id="pdfdiv"]/div[5]/div/h2/font[2]/b/text()').extract()
		
		
		if score!=[]:
			if type(score)==list:
				score = score[0].replace('/10','')
			else:
				score.replace('/10','')

			if score:
				self.url2weight[url] = int(score)
			else:
				self.url2weight[url] = 1
		else:
			
			self.url2weight[url] = 1

		
		
	# 解析各筆網頁
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
	
	# 結束爬蟲，顯示品牌熱度
	def spider_closed(self, spider):
		print("END")
		print('得分:',self.frequency_count)