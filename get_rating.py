import json
import numpy as np
import os
from argparse import ArgumentParser
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import collections
import jieba
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

parser = ArgumentParser(description="矩陣分解搜尋熱度評分")
parser.add_argument("-o",dest = "brands",help="請輸入欲查詢之品牌列表，請以分號;分隔並以字串形式呈現，如'vidiu;livestream'")
args = parser.parse_args()
brand_names = args.brands.split(';')

class IMP_ClickHeat:
	def __init__(self,brand_names):
		self.brand_names = brand_names
		for filename in os.listdir('googlesearch'):
			if filename.endswith('.json'):
				file_path = filename

		with open('googlesearch/'+file_path, 'r') as f:
			self.web_pages = json.load(f)


		self.R = []
		self.corpus = []
		self.pageRank_score = []

		self.nmf_pageRank_counter = collections.defaultdict(float)
		self.pageRank_counter = collections.defaultdict(float)
		self.tfidf_pageRank_counter = collections.defaultdict(float)

	def get_score(self):
		self.pageRank_get_score()
		self.nmf_pageRank_get_score()
		self.tfidf_pageRank_get_score()

		self.plot_bar(self.pageRank_counter,'pageRank')
		self.plot_bar(self.nmf_pageRank_counter,'pageRank+nmf')
		self.plot_bar(self.tfidf_pageRank_counter,'pageRank+tfidf')


	def processDocument(self,data):
		# 處理網頁資料。
		document = ''
		for sentence in jieba.cut(data):
			filter_string = ''.join(list(filter(str.isalpha, sentence.lower())))
			document+=(filter_string+' ')
		return document

	def normal_count(self,tfidf):
		# 單純記各品牌在各網頁之數量，以及儲存各網頁之pageRank

		if tfidf:
			for web_page in self.web_pages:
				self.corpus.append(self.processDocument(web_page['important_sentence']))
				self.R.append([self.processDocument(web_page['important_sentence']).count(key_word) for key_word in self.brand_names])
				self.pageRank_score.append(web_page['score'])
		else:
			for web_page in self.web_pages:
				self.R.append([self.processDocument(web_page['important_sentence']).count(key_word) for key_word in self.brand_names])
				self.pageRank_score.append(web_page['score'])

	def setupR(self,tfidf = False):
		# 設置評分矩陣R，列為各網頁，欄則為各品牌，值為各品牌在各網頁之出現次數。
		if not self.R:
			self.normal_count(tfidf)

	def pageRank_get_score(self):
		# 以pageRank作為權重進行搜尋熱度計算。
		self.setupR()

		for row_index in range(len(self.R)):
			for key_word_index in range(len(self.brand_names)):

				self.pageRank_counter[self.brand_names[key_word_index]]+=(self.R[row_index][key_word_index]*self.pageRank_score[row_index])

	def nmf_pageRank_get_score(self):

		# 使用非負矩陣分解來找出具有隱性因子之評分矩陣
		from sklearn.decomposition import NMF

		self.setupR()
		R =  np.array(normalize(self.R, norm='l2'))

		model = NMF(n_components=2, init='random', random_state=0)
		W = model.fit_transform(R)
		H = model.components_

		rating_matrix = np.dot(W,H)

		for brand_index in range(len(brand_names)):
			
			self.nmf_pageRank_counter[brand_names[brand_index]] = np.sum(rating_matrix[:][brand_index].tolist())

	def tfidf_pageRank_get_score(self):
		self.setupR(tfidf = True)
		self.tfidf_transfer()

		for row_index in range(len(self.R)):
			for key_word_index in range(len(self.brand_names)):
				if self.word2index[self.brand_names[key_word_index]] == None:
					continue
				self.tfidf_pageRank_counter[self.brand_names[key_word_index]]+=(self.R[row_index][key_word_index]*self.pageRank_score[row_index]*self.tfidf_weight[row_index][self.word2index[self.brand_names[key_word_index]]])

	def tfidf_transfer(self):

		self.corpus = [self.processDocument(web_page['important_sentence']) for web_page in self.web_pages]

		vectorizer = CountVectorizer()# 詞頻矩陣
		transformer = TfidfTransformer()#各詞tfidf權重
		tfidf = transformer.fit_transform(vectorizer.fit_transform(self.corpus))#第一個fit_transform是計算tf-idf，第二個fit_transform是轉換為詞頻矩陣
		word = vectorizer.get_feature_names()#取得所有詞語
		self.tfidf_weight = tfidf.toarray()

		self.word2index = {}
		for key_word in self.brand_names:
			self.word2index[key_word] = word.index(key_word) if key_word in word else None

	def plot_bar(self,frequency_count,name):
		index = np.arange(len(frequency_count.keys()))
		plt.bar(index, frequency_count.values())
		plt.xlabel('Degree of ClickHeat', fontsize=1)
		plt.ylabel('Brand', fontsize=12)
		plt.xticks(index, frequency_count.keys(), fontsize=12, rotation=30)
		plt.title('ClickHeat - {0}'.format(name))
		plt.show()


class MF():
	
	def __init__(self,R, K = 2, alpha = 0.1, beta = 0.01, iterations = 2000):
		"""
		Perform matrix factorization to predict empty
		entries in a matrix.
		
		Arguments
		- R (ndarray)   : user-item rating matrix
		- K (int)       : number of latent dimensions
		- alpha (float) : learning rate
		- beta (float)  : regularization parameter
		"""
		

		self.R =  np.array(normalize(R, norm='l2'))
		self.num_users, self.num_items = self.R.shape
		self.K = K
		self.alpha = alpha
		self.beta = beta
		self.iterations = iterations

		

	def train(self):
		# Initialize user and item latent feature matrice
		self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
		self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
		
		# Initialize the biases
		self.b_u = np.zeros(self.num_users)
		self.b_i = np.zeros(self.num_items)
		self.b = np.mean(self.R[np.where(self.R != 0)])
		
		# Create a list of training samples
		self.samples = [
			(i, j, self.R[i, j])
			for i in range(self.num_users)
			for j in range(self.num_items)
			if self.R[i, j] > 0
		]
		
		# Perform stochastic gradient descent for number of iterations
		training_process = []
		for i in range(self.iterations):
			np.random.shuffle(self.samples)
			self.sgd()
			mse = self.mse()
			training_process.append((i, mse))
			if (i+1) % 10 == 0:
				print("Iteration: %d ; error = %.4f" % (i+1, mse))
		
		return training_process

	def mse(self):
		"""
		A function to compute the total mean square error
		"""
		xs, ys = self.R.nonzero()
		predicted = self.full_matrix()
		error = 0
		for x, y in zip(xs, ys):
			error += pow(self.R[x, y] - predicted[x, y], 2)
		return np.sqrt(error)

	def sgd(self):
		"""
		Perform stochastic graident descent
		"""
		for i, j, r in self.samples:
			# Computer prediction and error
			prediction = self.get_rating(i, j)
			e = (r - prediction)
			
			# Update biases
			self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
			self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])
			
			# Update user and item latent feature matrices
			self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
			self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

	def get_rating(self, i, j):
		"""
		Get the predicted rating of user i and item j
		"""
		prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
		return prediction
	
	def full_matrix(self):
		"""
		Computer the full matrix using the resultant biases, P and Q
		"""
		return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)

if __name__ == '__main__':
	imp = IMP_ClickHeat(brand_names)
	imp.get_score()
	


