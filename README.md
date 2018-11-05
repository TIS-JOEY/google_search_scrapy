# google_search_scrapy
### 簡介
依所要查詢之google關鍵字來去查找特定品牌之搜尋熱度，以Google的Page Rank來作為網頁中關鍵字的權重，最後產生各個相關品牌的搜尋排行。

### 安裝
```
git clone https://github.com/TIS-JOEY/google_search_scrapy.git
```
```
$ cd google_search_scrapy
$ pip install -r requirements.txt
```

### Usage

註：iplist.txt檔為存儲代理IP的檔案。
```
$ cd googlesearch

# quires為所要查詢的詞，若要查詢多個詞請以分號;隔開。
# brandname為所要計算的品牌名，若有多個請以分號;隔開。
$ scrapy crawl googlesearchspider -a queries='livebox;livestream' -a brandname='vidiu;livestream;beam;tricaster;liveshell;'
```


