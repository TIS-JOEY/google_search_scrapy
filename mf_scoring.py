import json
import numpy as np
import os
from argparse import ArgumentParser
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

parser = ArgumentParser(description="矩陣分解搜尋熱度評分")
parser.add_argument("-o",dest = "brands",help="請輸入欲查詢之品牌列表，請以分號;分隔並以字串形式呈現，如'vidiu;livestream'")
args = parser.parse_args()
brand_names = args.brands.split(';')

class MF():
	
	def __init__(self, K = 2, alpha = 0.1, beta = 0.01, iterations = 2000):
		"""
		Perform matrix factorization to predict empty
		entries in a matrix.
		
		Arguments
		- R (ndarray)   : user-item rating matrix
		- K (int)       : number of latent dimensions
		- alpha (float) : learning rate
		- beta (float)  : regularization parameter
		"""
		
		for filename in os.listdir('googlesearch'):
			if filename.endswith('.json'):
				file_path = filename

		with open('googlesearch/'+file_path, 'r') as f:
			data = json.load(f)

		R = []
		for web_page in data:
			tmp_list = [web_page['important_sentence'].count(brand) for brand in brand_names]
			R.append(tmp_list)

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

	def plot_bar(self):
		rating_matrix = mf.full_matrix()

		frequency_count = {}
		for brand_index in range(len(brand_names)):
			frequency_count[brand_names[brand_index]] = np.mean((rating_matrix[:][brand_index]),0)

		index = np.arange(len(frequency_count.keys()))
		plt.bar(index, frequency_count.values())
		plt.xlabel('Degree of ClickHeat', fontsize=1)
		plt.ylabel('Brand', fontsize=12)
		plt.xticks(index, frequency_count.keys(), fontsize=12, rotation=30)
		plt.title('ClickHeat')
		plt.show()

if __name__ == '__main__':
	mf = MF()
	mf.train()
	mf.plot_bar()
	


