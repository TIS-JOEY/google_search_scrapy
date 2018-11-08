# google_search_scrapy
## Abstract
現有的Google Trend僅能處理查詢之關鍵字的搜尋熱度，但無法查知品牌在該產品中的隱性熱度為何。
因此，我們將Google Trend所得到的搜尋熱度作為顯性搜尋熱度。
而對於隱性搜尋熱度，則依所要查詢之google關鍵字產品進行Google搜尋爬文，並以Google的Page Rank來作為網頁中關鍵字的權重，最後產生各個相關品牌的隱性搜尋熱度排行。

### Require
請申請Plotly帳戶，並至googleTrend.py中填寫username與API KEY。


### Installation
```
git clone https://github.com/TIS-JOEY/google_search_scrapy.git
```
```
$ cd google_search_scrapy
$ pip install -r requirements.txt
```

## Usage
## 顯性搜尋熱度

-o： 請輸入欲查詢之關鍵字列表，請以分號;分隔。

-c：請輸入欲查詢之類別對應碼，https://github.com/pat310/google-trends-api/wiki/Google-Trends-Categories

-d：請輸入欲查詢之期間，以YYYY-MM-DD編寫，起始日期與結束日期請以空格分隔，舉例："2016-12-14 2017-01-25"，若要選擇所有期間，請編寫all。


```
python googleTrend.py -o 'vidiu;beam;livestream' -c 612 -d '2016-01-01 2018-11-08'
```
## 隱性搜尋熱度
註：iplist.txt檔為存儲代理IP的檔案。

-a quires：所要查詢的詞，若要查詢多個詞請以分號;隔開。
-a brandname：所要計算的品牌名，若有多個請以分號;隔開。
-o 檔名.json：儲存爬文資料的檔案。
```
$ cd googlesearch

$ scrapy crawl googlesearchspider -a queries='livebox;livestream' -a brandname='vidiu;livestream;beam;tricaster;liveshell' -o crawl.json
```


