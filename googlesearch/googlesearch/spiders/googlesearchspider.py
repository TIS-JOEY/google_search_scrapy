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

# 處理網頁資料
import re
import w3lib

# 其他
import numpy as np
import matplotlib.pyplot as plt
from scrapy.utils.misc import arg_to_iter
import json
import jieba
import jieba.posseg as pseg
import os
import sys
import random
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


class GooglesearchspiderSpider(scrapy.Spider):
	
	# 識別名，在terminal中填寫scrapy crawl googlesearchspider
	name = 'googlesearchspider'
	allowed_domains = []
	
	base_url_fmt = 'https://www.google.com.tw/search?q={query}'

	def __init__(self,queries,brandname):
		# 查詢詞列表
		self.queries = queries.split(';')

		# 欲計算之品牌
		self.normal_frequency_count = {brand:0 for brand in brandname.split(';')}
		self.tfidf_frequency_count = {brand:0 for brand in brandname.split(';')}

		# PageRank權重
		self.url2weight = {}

		# Base URL
		self.base_url = 'https://www.google.com.tw'

		dispatcher.connect(self.spider_closed, signals.spider_closed)

		for filename in os.listdir():
			if filename.endswith('.json'):
				os.remove(filename)

	def start_requests(self):

		# 依序將查詢詞投入Google搜尋並進行解析
		for query in arg_to_iter(self.queries):
			url = self.make_google_search_request(query)
			
			yield Request(url=url,callback = self.parse)

	def make_google_search_request(self, query):

		# 製作Google查詢之URL
		return self.base_url_fmt.format(query='+'.join(query.split()).strip('+'))

	def frequency_renew(self,key_word,doc,score,tfidf = False):

		# 計算品牌熱度
		doc = doc.lower()

		if tfidf:
			self.tfidf_frequency_count[key_word]+=(doc.count(key_word)*score)
		else:
			self.normal_frequency_count[key_word]+=(doc.count(key_word)*score)

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
				self.url2weight[url] = float(score)
			else:
				self.url2weight[url] = 1.0
		else:
			
			self.url2weight[url] = 1.0

		
		
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

		all_link = ' '.join(sel.xpath('*//a/@href').extract())
		all_h = ' '.join(sel.xpath('//h1/text()').extract())+' '.join(sel.xpath('//h2/text()').extract())+' '.join(sel.xpath('//h3/text()').extract())
		all_p = ' '.join(sel.xpath('//p/text()').extract())
		
		pattern = re.compile(r"\s+")

		html = pattern.sub(" ", html)
		important_sentence = pattern.sub(" ",all_link+" "+all_h+" "+all_p+" "+name)

		item['name'] = name
		item['description'] = html
		item['important_sentence'] = important_sentence
		item['url'] = url		
		item['score'] = self.url2weight[url]

		#list(map(lambda key_word:self.frequency_renew(key_word,item['important_sentence'],item['score']), self.normal_frequency_count))
		
		yield item
	
	
	# this is for plotting purpose
	def plot_bar(self,frequency_count):
	    index = np.arange(len(frequency_count.keys()))
	    plt.bar(index, frequency_count.values())
	    plt.xlabel('Degree of ClickHeat', fontsize=1)
	    plt.ylabel('Brand', fontsize=12)
	    plt.xticks(index, frequency_count.keys(), fontsize=12, rotation=30)
	    plt.title('ClickHeat')
	    plt.show()

	def plotly_bar(self,frequency_count):
		import plotly.plotly as py
		import plotly.graph_objs as go
		import plotly

		plotly.tools.set_credentials_file(username='TIS-JOEY', api_key='jU8KZYrSWISLzElR3x3v')

		data = []
		for key,value in frequency_count.items():
			data.append(go.Bar(
			            x=[key],
			            y=[value],
			            text=[value],
			            textposition = 'auto',
			            marker= dict(color='rgb({0},{1},{2})'.format(random.randint(1,255),random.randint(1,255),random.randint(1,255)),line=dict(color='rgb(8,48,107)',width=1.5),),
			            opacity=0.6
			        ))

		py.iplot(data, filename='bar-direct-labels')

	def _tfidf_transfer(self,data):
		document = ''
		jieba.add_word('livestream box')
		for sentence in jieba.cut(data['important_sentence']):
			filter_string = ''.join(list(filter(str.isalpha, sentence.lower())))
			document+=(filter_string+' ')
		self.pageRank_score.append(data['score'])
		return document

	def tfidf_transfer(self):

		for filename in os.listdir():
			if filename.endswith('.json'):
				corpus_path = filename

		self.pageRank_score = []
		with open(corpus_path,'r') as f:
			datas = json.load(f)

		self.corpus = list(map(self._tfidf_transfer,datas))
		

		vectorizer = CountVectorizer()# 詞頻矩陣
		transformer = TfidfTransformer()#各詞tfidf權重
		tfidf = transformer.fit_transform(vectorizer.fit_transform(self.corpus))#第一個fit_transform是計算tf-idf，第二個fit_transform是轉換為詞頻矩陣
		word = vectorizer.get_feature_names()#取得所有詞語
		self.tfidf_weight = tfidf.toarray()

		self.word2index = {}
		for key_word in self.tfidf_frequency_count:
			self.word2index[key_word] = self.word.index(key_word) if key_word in self.word else 0

	def tfidf_get_score(self):
		self.tfidf_transfer()
		for document_index in range(len(self.corpus)):
			list(map(lambda key_word:self.frequency_renew(key_word,self.corpus[document_index],self.pageRank_score[document_index]*self.tfidf_weight[document_index][self.word2index[key_word]] ,tfidf = True), self.tfidf_frequency_count))

	# 結束爬蟲，顯示品牌熱度
	def spider_closed(self, spider):
		print("END")
		#print('pageRank得分:',self.normal_frequency_count)
		#self.plotly_bar(self.normal_frequency_count)
		#self.plot_bar(self.normal_frequency_count)
		#print('pageRank+tfidf得分:',self.tfidf_frequency_count)
		#self.plotly_bar(self.tfidf_frequency_count)
		#self.plot_bar(self.tfidf_frequency_count)

