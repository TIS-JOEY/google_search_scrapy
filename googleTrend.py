from pytrends.request import TrendReq
import plotly.plotly as py
import plotly.graph_objs as go
from argparse import ArgumentParser
import plotly

plotly.tools.set_credentials_file(username='your username', api_key='your api key')


parser = ArgumentParser(description="Google搜尋品牌熱度")
parser.add_argument("-o",dest = "keyword_list",help="請輸入欲查詢之關鍵字列表，請以分號;分隔")
parser.add_argument("-c",dest = "cat",help = '請輸入欲查詢之類別對應碼，https://github.com/pat310/google-trends-api/wiki/Google-Trends-Categories')
parser.add_argument("-d",dest = "date",help = '請輸入欲查詢之期間，以YYYY-MM-DD編寫，起始日期與結束日期請以空格分隔，舉例："2016-12-14 2017-01-25"，若要選擇所有期間，請編寫all')
args = parser.parse_args()

keyword_list = args.keyword_list.split(';')
cat = args.cat
date = args.date

plot_list = []
for keyword in keyword_list:
	kw_list = [keyword]
	pytrends = TrendReq(proxies = {'https': 'https://194.186.180.118:61837'})

	pytrends.build_payload(kw_list, cat=612, timeframe=date)

	df = pytrends.interest_over_time()

	plot_list.append(go.Scatter(x = list(map(lambda x:str(x)[:10],df.index.values)),y = df.iloc[:,0],mode = 'lines',name = keyword))

py.iplot(plot_list, filename='line-mode')


