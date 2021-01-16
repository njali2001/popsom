import sys
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import seaborn as sns
import statsmodels.stats.api as sms     # t-test
import statistics as stat
from scipy import stats 
from random import randint
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter
from scipy.stats import f 
import matplotlib.pyplot as plt

class map:
	def __init__(self, xdim=10, ydim=5, alpha=.3, train=1000, normalize=True, seed=None):
		self.xdim = xdim
		self.ydim = ydim
		self.alpha = alpha
		self.train = train
		self.normalize = normalize
		self.seed = seed

	@property
	def xdim(self):
		return self._xdim

	@xdim.setter
	def xdim(self, value):
		if value < 5:
			raise ValueError("map: map is too small.")
		self._xdim = value

	@property
	def ydim(self):
		return self._ydim

	@ydim.setter
	def ydim(self, value):
		if value < 5:
			raise ValueError("map: map is too small.")
		self._ydim = value

	@property
	def train(self):
		return self._train

	@train.setter
	def train(self, value):
		if (isinstance(value, int) and  (value <= 0)): 
			raise Exception("map: seed value has to be a positive integer value")
		self._train = value

	@property
	def seed(self):
		return self._seed

	@seed.setter
	def seed(self, value):
		if (isinstance(value, int) and  (value <= 0)): 
			raise Exception("map: seed value has to be a positive integer value")
		self._seed = value


	def fit(self, data, labels):

		if isinstance(data, pd.DataFrame):	
			for column in data:
				if not is_numeric_dtype(data[column]):
					raise ValueError("map: only numeric data can be used for training")		
			if self.normalize:
				self.data = data.div(data.sum(axis=1), axis=0)
			else:
				self.data = data
		else:
			raise ValueError("map: training data has to be a data frame")

		self.labels = labels

		self.vsom_p()

		self.compute_heat()

		self.map_fitted_obs()

		self.compute_centroids()

		self.get_unique_centroids()
		
		self.majority_labels()

		self.compute_label_to_centroid()

		self.compute_centroid_obs()

		self.map_convergence()

		self.compute_wcss()

		self.compute_bcss()

	def vsom_p(self):
		""" vsom_p -- vectorized, unoptimized version of the stochastic SOM
        		 	  training algorithm written entirely in python
    	"""
    	# some constants
		dr = self.data.shape[0]
		dc = self.data.shape[1]
		nr = self.xdim*self.ydim
		nc = dc  # dim of data and neurons is the same

	    # build and initialize the matrix holding the neurons
		cells = nr * nc  # No. of neurons times number of data dimensions

	    # vector with small init values for all neurons
		v = np.random.uniform(-1, 1, cells)

	    # NOTE: each row represents a neuron, each column represents a dimension.
		neurons = np.transpose(np.reshape(v, (nc, nr)))  # rearrange the vector as matrix

		# neurons = np.reshape(v, (nr, nc)) # Another option to reshape

	    # compute the initial neighborhood size and step
		nsize = max(self.xdim, self.ydim) + 1
		nsize_step = np.ceil(self.train/nsize)
		step_counter = 0  # counts the number of epochs per nsize_step

	    # convert a 1D rowindex into a 2D map coordinate
		def coord2D(rowix):

			x = np.array(rowix) % self.xdim
			y = np.array(rowix) // self.xdim

			return np.concatenate((x, y))

	    # constants for the Gamma function
		m = [i for i in range(nr)]  # a vector with all neuron 1D addresses

	    # x-y coordinate of ith neuron: m2Ds[i,] = c(xi, yi)
		m2Ds = np.matrix.transpose(coord2D(m).reshape(2, nr))

	    # neighborhood function
		def Gamma(c):

	        # lookup the 2D map coordinate for c
			c2D = m2Ds[c, ]
	        # a matrix with each row equal to c2D
			c2Ds = np.outer(np.linspace(1, 1, nr), c2D)
	        # distance vector of each neuron from c in terms of map coords!
			d = np.sqrt(np.dot((c2Ds - m2Ds)**2, [1, 1]))
	        # if m on the grid is in neigh then alpha else 0.0
			hood = np.where(d < nsize*1.5, self.alpha, 0.0)

			return hood
		
	    # training #
	    # the epochs loop
		
		for epoch in range(self.train):

	        # hood size decreases in disrete nsize.steps
			step_counter = step_counter + 1
			if step_counter == nsize_step:

				step_counter = 0
				nsize = nsize - 1

	        # create a sample training vector
			ix = randint(0, dr-1)
			# ix = (epoch+1) % dr   # For Debugging
			xk = self.data.iloc[[ix]]

	        # competitive step
			xk_m = np.outer(np.linspace(1, 1, nr), xk)
			
			diff = neurons - xk_m
			squ = diff * diff
			s = np.dot(squ, np.linspace(1, 1, nc))
			o = np.argsort(s)
			c = o[0]

	        # update step
			gamma_m = np.outer(Gamma(c), np.linspace(1, 1, nc))
			neurons = neurons - diff * gamma_m
		
		self.neurons = neurons

	def compute_heat(self):
	
		d = euclidean_distances(self.neurons, self.neurons)
		x = self.xdim
		y = self.ydim
		heat = np.matrix([[0.0] * y for _ in range(x)])

		if x == 1 or y == 1:
			sys.exit("compute_heat: heat map can not be computed for a map \
	                 with a dimension of 1")

		# this function translates our 2-dim map coordinates
		# into the 1-dim coordinates of the neurons
		def xl(ix, iy):

			return ix + iy * x

		# check if the map is larger than 2 x 2 (otherwise it is only corners)
		if x > 2 and y > 2:
			# iterate over the inner nodes and compute their umat values
			for ix in range(1, x-1):
				for iy in range(1, y-1):
					sum = (d[xl(ix, iy), xl(ix-1, iy-1)] +
						   d[xl(ix, iy), xl(ix, iy-1)] +
	                       d[xl(ix, iy), xl(ix+1, iy-1)] +
	                       d[xl(ix, iy), xl(ix+1, iy)] +
	                       d[xl(ix, iy), xl(ix+1, iy+1)] +
	                       d[xl(ix, iy), xl(ix, iy+1)] +
	                       d[xl(ix, iy), xl(ix-1, iy+1)] +
	                       d[xl(ix, iy), xl(ix-1, iy)])

					heat[ix, iy] = sum/8

			# iterate over bottom x axis
			for ix in range(1, x-1):
				iy = 0
				sum = (d[xl(ix, iy), xl(ix+1, iy)] +
	                   d[xl(ix, iy), xl(ix+1, iy+1)] +
	                   d[xl(ix, iy), xl(ix, iy+1)] +
	                   d[xl(ix, iy), xl(ix-1, iy+1)] +
	                   d[xl(ix, iy), xl(ix-1, iy)])

				heat[ix, iy] = sum/5

			# iterate over top x axis
			for ix in range(1, x-1):
				iy = y-1
				sum = (d[xl(ix, iy), xl(ix-1, iy-1)] +
	                   d[xl(ix, iy), xl(ix, iy-1)] +
	                   d[xl(ix, iy), xl(ix+1, iy-1)] +
	                   d[xl(ix, iy), xl(ix+1, iy)] +
	                   d[xl(ix, iy), xl(ix-1, iy)])

				heat[ix, iy] = sum/5

			# iterate over the left y-axis
			for iy in range(1, y-1):
				ix = 0
				sum = (d[xl(ix, iy), xl(ix, iy-1)] +
	                   d[xl(ix, iy), xl(ix+1, iy-1)] +
	                   d[xl(ix, iy), xl(ix+1, iy)] +
	                   d[xl(ix, iy), xl(ix+1, iy+1)] +
	                   d[xl(ix, iy), xl(ix, iy+1)])

				heat[ix, iy] = sum/5

			# iterate over the right y-axis
			for iy in range(1, y-1):
				ix = x-1
				sum = (d[xl(ix, iy), xl(ix-1, iy-1)] +
	                   d[xl(ix, iy), xl(ix, iy-1)] +
	                   d[xl(ix, iy), xl(ix, iy+1)] +
	                   d[xl(ix, iy), xl(ix-1, iy+1)] +
	                   d[xl(ix, iy), xl(ix-1, iy)])

				heat[ix, iy] = sum/5

		# compute umat values for corners
		if x >= 2 and y >= 2:
			# bottom left corner
			ix = 0
			iy = 0
			sum = (d[xl(ix, iy), xl(ix+1, iy)] +
	               d[xl(ix, iy), xl(ix+1, iy+1)] +
	               d[xl(ix, iy), xl(ix, iy+1)])

			heat[ix, iy] = sum/3

			# bottom right corner
			ix = x-1
			iy = 0
			sum = (d[xl(ix, iy), xl(ix, iy+1)] +
	               d[xl(ix, iy), xl(ix-1, iy+1)] +
	               d[xl(ix, iy), xl(ix-1, iy)])
			heat[ix, iy] = sum/3

			# top left corner
			ix = 0
			iy = y-1
			sum = (d[xl(ix, iy), xl(ix, iy-1)] +
	               d[xl(ix, iy), xl(ix+1, iy-1)] +
	               d[xl(ix, iy), xl(ix+1, iy)])
			heat[ix, iy] = sum/3

			# top right corner
			ix = x-1
			iy = y-1
			sum = (d[xl(ix, iy), xl(ix-1, iy-1)] +
	               d[xl(ix, iy), xl(ix, iy-1)] +
	               d[xl(ix, iy), xl(ix-1, iy)])
			heat[ix, iy] = sum/3

		# smooth the heat map
		pts = []

		for i in range(y):
			for j in range(x):
				pts.extend([[j, i]])

			heat = self.smooth_2d(heat,
								  nrow=x,
								  ncol=y,
								  surface=False,
								  theta=2)

		# Only for test
		heat = np.array([
			[10.894488,12.067031,13.592042,14.363678,13.363055,11.327550,9.874710,9.450540,9.413375,9.295735],
			[11.233770,12.256429,13.635697,14.385219,13.511998,11.620058,10.251016,9.933695,10.088685,10.196684],
			[11.352180,12.134874,13.240565,13.891809,13.229272,11.687812,10.576612,10.417132,10.728264,11.006791],
			[11.191980,11.611879,12.243484,12.624846,12.203021,11.244226,10.619160,10.676830,11.052352,11.362853],
			[10.733324,10.756098,10.827880,10.837374,10.632647,10.335093,10.271935,10.509745,10.800660,10.962590],
			[9.751070,9.574864,9.330125,9.142441,9.149330,9.347178,9.629504,9.846281,9.872044,9.732489],
			[8.349649,8.284139,8.129459,8.055654,8.264111,8.649540,8.920836,8.898770,8.582514,8.130640],
			[7.392189,7.572249,7.693369,7.853830,8.183262,8.518838,8.576746,8.277604,7.724574,7.112298],
			[7.694719,8.034522,8.332381,8.600886,8.901384,9.098433,8.995074,8.590177,8.019210,7.449536],
			[9.122158,9.516873,9.873250,10.088783,10.154954,10.093539,9.919880,9.652167,9.313255,8.968774],
			[10.675677,11.130425,11.594275,11.760161,11.434118,10.879503,10.540330,10.501070,10.536964,10.511776],
			[11.300875,11.897156,12.635223,12.952030,12.320442,11.178213,10.505420,10.565858,10.887557,11.103263],
			[10.736490,11.555977,12.699824,13.393607,12.742778,11.209174,10.199810,10.197460,10.591644,10.863165],
			[9.590106,10.649877,12.191234,13.311470,12.838938,11.173372,9.959033,9.839458,10.159812,10.349565],
			[8.543830,9.789611,11.615839,13.054133,12.769697,11.095411,9.781385,9.557985,9.764729,9.833379]])


		self.heat = heat

	def smooth_2d(self, Y, ind=None, weight_obj=None, grid=None, nrow=64, ncol=64, surface=True, theta=None):
		""" smooth_2d -- Kernel Smoother For Irregular 2-D Data """

		def exp_cov(x1, x2, theta=2, p=2, distMat=0):
			x1 = x1*(1/theta)
			x2 = x2*(1/theta)
			distMat = euclidean_distances(x1, x2)
			distMat = distMat**p
			return np.exp(-distMat)

		NN = [[1]*ncol] * nrow
		grid = {'x': [i for i in range(nrow)], "y": [i for i in range(ncol)]}

		if weight_obj is None:
			dx = grid['x'][1] - grid['x'][0]
			dy = grid['y'][1] - grid['y'][0]
			m = len(grid['x'])
			n = len(grid['y'])
			M = 2 * m
			N = 2 * n
			xg = []

			for i in range(N):
				for j in range(M):
					xg.extend([[j, i]])

			xg = np.matrix(xg)

			center = []
			center.append([int(dx * M/2-1), int((dy * N)/2-1)])

			out = exp_cov(xg, np.matrix(center),theta=theta)
			out = np.matrix.transpose(np.reshape(out, (N, M)))
			temp = np.zeros((M, N))
			temp[int(M/2-1)][int(N/2-1)] = 1

			wght = np.fft.fft2(out)/(np.fft.fft2(temp) * M * N)
			weight_obj = {"m": m, "n": n, "N": N, "M": M, "wght": wght}

		temp = np.zeros((weight_obj['M'], weight_obj['N']))
		temp[0:m, 0:n] = Y
		temp2 = np.fft.ifft2(np.fft.fft2(temp) *
							 weight_obj['wght']).real[0:weight_obj['m'],
													  0:weight_obj['n']]

		temp = np.zeros((weight_obj['M'], weight_obj['N']))
		temp[0:m, 0:n] = NN
		temp3 = np.fft.ifft2(np.fft.fft2(temp) *
							 weight_obj['wght']).real[0:weight_obj['m'],
													  0:weight_obj['n']]

		return temp2/temp3

	def map_fitted_obs(self):
		fitted_obs =[]
		for i in range(self.data.shape[0]):
			b = self.best_match(self.data.iloc[[i]])
			fitted_obs.extend([b])

		self.fitted_obs = fitted_obs

	def best_match(self, obs, full=False):
		""" best_match -- given observation obs, return the best matching neuron """

	   	# NOTE: replicate obs so that there are nr rows of obs
		obs_m = np.tile(obs, (self.neurons.shape[0], 1))
		diff = self.neurons - obs_m
		squ = diff * diff
		s = np.sum(squ, axis=1)
		d = np.sqrt(s)
		o = np.argsort(d)

		if full:
			return o
		else:
			return o[0]

	def compute_centroids(self):

		heat = np.matrix(self.heat)
		xdim = self.xdim
		ydim = self.ydim
		centroids = [ [ {'x':-1,'y':-1} for i in range(ydim) ] for j in range(xdim) ]

		def compute_centroid(ix, iy):
			
			if (centroids[ix][iy]['x'] > -1) and (centroids[ix][iy]['y'] > -1):
				return centroids[ix][iy]

			min_val = heat[ix, iy]
			min_x = ix
			min_y = iy

			# (ix, iy) is an inner map element
			if ix > 0 and ix < xdim-1 and iy > 0 and iy < ydim-1:

				if heat[ix-1, iy-1] < min_val:
					min_val = heat[ix-1, iy-1]
					min_x = ix-1
					min_y = iy-1

				if heat[ix, iy-1] < min_val:
					min_val = heat[ix, iy-1]
					min_x = ix
					min_y = iy-1

				if heat[ix+1, iy-1] < min_val:
					min_val = heat[ix+1, iy-1]
					min_x = ix+1
					min_y = iy-1

				if heat[ix+1, iy] < min_val:
					min_val = heat[ix+1, iy]
					min_x = ix+1
					min_y = iy

				if heat[ix+1, iy+1] < min_val:
					min_val = heat[ix+1, iy+1]
					min_x = ix+1
					min_y = iy+1

				if heat[ix, iy+1] < min_val:
					min_val = heat[ix, iy+1]
					min_x = ix
					min_y = iy+1

				if heat[ix-1, iy+1] < min_val:
					min_val = heat[ix-1, iy+1]
					min_x = ix-1
					min_y = iy+1

				if heat[ix-1, iy] < min_val:
					min_val = heat[ix-1, iy]
					min_x = ix-1
					min_y = iy

			# (ix, iy) is bottom left corner
			elif ix == 0 and iy == 0:

				if heat[ix+1, iy] < min_val:
					min_val = heat[ix+1, iy]
					min_x = ix+1
					min_y = iy

				if heat[ix+1, iy+1] < min_val:
					min_val = heat[ix+1, iy+1]
					min_x = ix+1
					min_y = iy+1

				if heat[ix, iy+1] < min_val:
					min_val = heat[ix, iy+1]
					min_x = ix
					min_y = iy+1

			# (ix, iy) is bottom right corner
			elif ix == xdim-1 and iy == 0:

				if heat[ix, iy+1] < min_val:
					min_val = heat[ix, iy+1]
					min_x = ix
					min_y = iy+1

				if heat[ix-1, iy+1] < min_val:
					min_val = heat[ix-1, iy+1]
					min_x = ix-1
					min_y = iy+1

				if heat[ix-1, iy] < min_val:
					min_val = heat[ix-1, iy]
					min_x = ix-1
					min_y = iy

			# (ix, iy) is top right corner
			elif ix == xdim-1 and iy == ydim-1:

				if heat[ix-1, iy-1] < min_val:
					min_val = heat[ix-1, iy-1]
					min_x = ix-1
					min_y = iy-1

				if heat[ix, iy-1] < min_val:
					min_val = heat[ix, iy-1]
					min_x = ix
					min_y = iy-1

				if heat[ix-1, iy] < min_val:
					min_val = heat[ix-1, iy]
					min_x = ix-1
					min_y = iy

			# (ix, iy) is top left corner
			elif ix == 0 and iy == ydim-1:

				if heat[ix, iy-1] < min_val:
					min_val = heat[ix, iy-1]
					min_x = ix
					min_y = iy-1

				if heat[ix+1, iy-1] < min_val:
					min_val = heat[ix+1, iy-1]
					min_x = ix+1
					min_y = iy-1

				if heat[ix+1, iy] < min_val:
					min_val = heat[ix+1, iy]
					min_x = ix+1
					min_y = iy

			# (ix, iy) is a left side element
			elif ix == 0 and iy > 0 and iy < ydim-1:

				if heat[ix, iy-1] < min_val:
					min_val = heat[ix, iy-1]
					min_x = ix
					min_y = iy-1

				if heat[ix+1, iy-1] < min_val:
					min_val = heat[ix+1, iy-1]
					min_x = ix+1
					min_y = iy-1

				if heat[ix+1, iy] < min_val:
					min_val = heat[ix+1, iy]
					min_x = ix+1
					min_y = iy

				if heat[ix+1, iy+1] < min_val:
					min_val = heat[ix+1, iy+1]
					min_x = ix+1
					min_y = iy+1

				if heat[ix, iy+1] < min_val:
					min_val = heat[ix, iy+1]
					min_x = ix
					min_y = iy+1

			# (ix, iy) is a bottom side element
			elif ix > 0 and ix < xdim-1 and iy == 0:

				if heat[ix+1, iy] < min_val:
					min_val = heat[ix+1, iy]
					min_x = ix+1
					min_y = iy
	
				if heat[ix+1, iy+1] < min_val:
					min_val = heat[ix+1, iy+1]
					min_x = ix+1
					min_y = iy+1

				if heat[ix, iy+1] < min_val:
					min_val = heat[ix, iy+1]
					min_x = ix
					min_y = iy+1

				if heat[ix-1, iy+1] < min_val:
					min_val = heat[ix-1, iy+1]
					min_x = ix-1
					min_y = iy+1

				if heat[ix-1, iy] < min_val:
					min_val = heat[ix-1, iy]
					min_x = ix-1
					min_y = iy

			# (ix, iy) is a right side element
			elif ix == xdim-1 and iy > 0 and iy < ydim-1:

				if heat[ix-1, iy-1] < min_val:
					min_val = heat[ix-1, iy-1]
					min_x = ix-1
					min_y = iy-1

				if heat[ix, iy-1] < min_val:
					min_val = heat[ix, iy-1]
					min_x = ix
					min_y = iy-1

				if heat[ix, iy+1] < min_val:
					min_val = heat[ix, iy+1]
					min_x = ix
					min_y = iy+1

				if heat[ix-1, iy+1] < min_val:
					min_val = heat[ix-1, iy+1]
					min_x = ix-1
					min_y = iy+1

				if heat[ix-1, iy] < min_val:
					min_val = heat[ix-1, iy]
					min_x = ix-1
					min_y = iy

			# (ix, iy) is a top side element
			elif ix > 0 and ix < xdim-1 and iy == ydim-1:

				if heat[ix-1, iy-1] < min_val:
					min_val = heat[ix-1, iy-1]
					min_x = ix-1
					min_y = iy-1

				if heat[ix, iy-1] < min_val:
					min_val = heat[ix, iy-1]
					min_x = ix
					min_y = iy-1

				if heat[ix+1, iy-1] < min_val:
					min_val = heat[ix+1, iy-1]
					min_x = ix+1
					min_y = iy-1

				if heat[ix+1, iy] < min_val:
					min_val = heat[ix+1, iy]
					min_x = ix+1
					min_y = iy

				if heat[ix-1, iy] < min_val:
					min_val = heat[ix-1, iy]
					min_x = ix-1
					min_y = iy


			if (min_x != ix) or (min_y != iy):
				return compute_centroid(min_x, min_y)
			else:
				return {'x': ix, 'y': iy}

		for i in range(xdim):
			for j in range(ydim):
				centroids[i][j] = compute_centroid(i, j)

		self.centroids = centroids

	def get_unique_centroids(self):
		centroids = self.centroids
		xdim = self.xdim
		ydim = self.ydim
		cd_list = []

		for ix in range(xdim):
			for iy in range(ydim):
				c_xy = centroids[ix][iy]
				if c_xy not in cd_list:
					cd_list.append(c_xy)
	
		self.unique_centroids = cd_list	      

	def majority_labels(self):
		x = self.xdim
		y = self.ydim
		centroids = self.centroids
		nobs = self.data.shape[0]

		centroid_labels =  [ [ None for i in range(y) ] for j in range(x) ]  
		majority_labels =  [ [ None for i in range(y) ] for j in range(x) ]  

		for i in range(nobs):
			lab = self.labels.iloc[i][0] 
			nix = self.fitted_obs[i]
			c = self.coordinate(nix)
			ix = c['x']
			iy = c['y']
			cx = centroids[ix][iy]['x']
			cy = centroids[ix][iy]['y']
			if centroid_labels[cx][cy]:
				centroid_labels[cx][cy] = centroid_labels[cx][cy] + ' ' + lab
			else:
				centroid_labels[cx][cy] = lab

		for ix in range(x):
			for iy in range(y):
				if centroid_labels[ix][iy]:
					label_v = list(centroid_labels[ix][iy].split(" "))
					majority = Counter(label_v)
					if len(majority) == 1:
						majority_labels[ix][iy] = label_v[0]
					else:
						majority_labels[ix][iy] = max(set(label_v), key = label_v.count)

		self.centroid_labels = majority_labels

	def compute_label_to_centroid(self):  
		conv ={}

		for i in range(len(self.unique_centroids)):
			x = self.unique_centroids[i]['x']
			y = self.unique_centroids[i]['y']
			l = self.centroid_labels[x][y]
			if l in conv:
				conv[l] += ' ' + str(i)
			else:
				conv[l] = str(i)

		self.label_to_centroid = conv

	def compute_centroid_obs(self):

		centroid_obs =  [ [] for i in range(len(self.unique_centroids)) ] 

		for cluster_ix in range(len(self.unique_centroids)):
			c_nix = self.rowix(self.unique_centroids[cluster_ix])
			for i in range(self.data.shape[0]):
				coord = self.coordinate(self.fitted_obs[i])
				c_obj_nix = self.rowix(self.centroids[coord['x']][coord['y']])
				if (c_obj_nix == c_nix):
					centroid_obs[cluster_ix].append(i)

		self.centroid_obs = centroid_obs

	def map_convergence(self, conf_int=.95, k=50, verb=False, ks=True):
	
		if ks:
			embed = self.embed_ks(conf_int, verb=False)
		else:
			embed = self.embed_vm(conf_int, verb=False)

		topo_ = self.topo(k, conf_int, verb=False, interval=False)

		if verb:
			self.convergence = {"embed": embed, "topo": topo_}
		else:
			self.convergence = (0.5*embed + 0.5*topo_)		

	def embed_ks(self, conf_int=0.95, verb=False):
		""" embed_ks -- using the kolgomorov-smirnov test """

		# map_df is a dataframe that contains the neurons
		map_df = self.neurons

		# data_df is a dataframe that contain the training data
		data_df = np.array(self.data)

		nfeatures = map_df.shape[1]

		# use the Kolmogorov-Smirnov Test to test whether the neurons and training
		# data appear
		# to come from the same distribution
		ks_vector = []
		for i in range(nfeatures):
			ks_vector.append(stats.mstats.ks_2samp(map_df[:, i], data_df[:, i]))

		prob_v = self.significance(graphics=False)
		var_sum = 0

		# compute the variance captured by the map
		for i in range(nfeatures):

			# the second entry contains the p-value
			if ks_vector[i][1] > (1 - conf_int):
				var_sum = var_sum + prob_v[i]
			else:
				# not converged - zero out the probability
				prob_v[i] = 0

		# return the variance captured by converged features
		if verb:
			return prob_v
		else:
			return var_sum

	def embed_vm(self, conf_int=.95, verb=False):
		""" embed_vm -- using variance and mean tests  """

		# map_df is a dataframe that contains the neurons
		map_df = self.neurons

		# data_df is a dataframe that contain the training data
		data_df = np.array(self.data)

		def df_var_test(df1, df2, conf=.95):

			if df1.shape[1] != df2.shape[1]:
				sys.exit("df_var_test: cannot compare variances of data frames")

			# init our working arrays
			var_ratio_v = [randint(1, 1) for _ in range(df1.shape[1])]
			var_confintlo_v = [randint(1, 1) for _ in range(df1.shape[1])]
			var_confinthi_v = [randint(1, 1) for _ in range(df1.shape[1])]

			def var_test(x, y, ratio=1, conf_level=0.95):

				DF_x = len(x) - 1
				DF_y = len(y) - 1
				V_x = stat.variance(x.tolist())
				V_y = stat.variance(y.tolist())

				ESTIMATE = V_x / V_y

				BETA = (1 - conf_level) / 2
				CINT = [ESTIMATE / f.ppf(1 - BETA, DF_x, DF_y),
						ESTIMATE / f.ppf(BETA, DF_x, DF_y)]

				return {"estimate": ESTIMATE, "conf_int": CINT}

		    # compute the F-test on each feature in our populations
			for i in range(df1.shape[1]):

				t = var_test(df1[:, i], df2[:, i], conf_level=conf)
				var_ratio_v[i] = t['estimate']
				var_confintlo_v[i] = t['conf_int'][0]
				var_confinthi_v[i] = t['conf_int'][1]

			# return a list with the ratios and conf intervals for each feature
			return {"ratio": var_ratio_v,
					"conf_int_lo": var_confintlo_v,
					"conf_int_hi": var_confinthi_v}

		def df_mean_test(df1, df2, conf=0.95):

			if df1.shape[1] != df2.shape[1]:
				sys.exit("df_mean_test: cannot compare means of data frames")

			# init our working arrays
			mean_diff_v = [randint(1, 1) for _ in range(df1.shape[1])]
			mean_confintlo_v = [randint(1, 1) for _ in range(df1.shape[1])]
			mean_confinthi_v = [randint(1, 1) for _ in range(df1.shape[1])]

			def t_test(x, y, conf_level=0.95):
				estimate_x = np.mean(x)
				estimate_y = np.mean(y)
				cm = sms.CompareMeans(sms.DescrStatsW(x), sms.DescrStatsW(y))
				conf_int_lo = cm.tconfint_diff(alpha=1-conf_level, usevar='unequal')[0]
				conf_int_hi = cm.tconfint_diff(alpha=1-conf_level, usevar='unequal')[1]

				return {"estimate": [estimate_x, estimate_y],
						"conf_int": [conf_int_lo, conf_int_hi]}

			# compute the F-test on each feature in our populations
			for i in range(df1.shape[1]):
				t = t_test(x=df1[:, i], y=df2[:, i], conf_level=conf)
				mean_diff_v[i] = t['estimate'][0] - t['estimate'][1]
				mean_confintlo_v[i] = t['conf_int'][0]
				mean_confinthi_v[i] = t['conf_int'][1]

			# return a list with the ratios and conf intervals for each feature
			return {"diff": mean_diff_v,
					"conf_int_lo": mean_confintlo_v,
					"conf_int_hi": mean_confinthi_v}
		# do the F-test on a pair of datasets
		vl = df_var_test(map_df, data_df, conf_int)

		# do the t-test on a pair of datasets
		ml = df_mean_test(map_df, data_df, conf=conf_int)

		# compute the variance captured by the map --
		# but only if the means have converged as well.
		nfeatures = map_df.shape[1]
		prob_v = self.significance(graphics=False)
		var_sum = 0

		for i in range(nfeatures):

			if (vl['conf_int_lo'][i] <= 1.0 and vl['conf_int_hi'][i] >= 1.0 and
				ml['conf_int_lo'][i] <= 0.0 and ml['conf_int_hi'][i] >= 0.0):

				var_sum = var_sum + prob_v[i]
			else:
				# not converged - zero out the probability
				prob_v[i] = 0

		# return the variance captured by converged features
		if verb:
			return prob_v
		else:
			return var_sum

	def topo(self, k=50, conf_int=.95, verb=False, interval=True):
		# data.df is a matrix that contains the training data
		data_df = self.data

		if (k > data_df.shape[0]):
			sys.exit("topo: sample larger than training data.")

		data_sample_ix = [randint(1, data_df.shape[0]) for _ in range(k)]

		# compute the sum topographic accuracy - the accuracy of a single sample
		# is 1 if the best matching unit is a neighbor otherwise it is 0
		acc_v = []
		for i in range(k):
			acc_v.append(self.accuracy(data_df.iloc[data_sample_ix[i]-1], data_sample_ix[i]))

		# compute the confidence interval values using the bootstrap
		if interval:
			bval = self.bootstrap(conf_int, data_df, k, acc_v)

		# the sum topographic accuracy is scaled by the number of samples -
		# estimated
		# topographic accuracy
		if verb:
			return acc_v
		else:
			val = np.sum(acc_v)/k
			if interval:
				return {'val': val, 'lo': bval['lo'], 'hi': bval['hi']}
			else:
				return val

	def compute_wcss(self):
		clusters_ss	= []
		for cluster_ix in range(len(self.unique_centroids)):
			c_nix = self.rowix(self.unique_centroids[cluster_ix])		
			vectors = self.neurons[c_nix,]
			for i in range(len(self.centroid_obs[cluster_ix])):
				obs_ix = self.centroid_obs[cluster_ix][i]
				vectors = np.vstack((vectors,self.data.iloc[obs_ix].to_numpy()))

			distances = euclidean_distances(vectors,vectors)[0]
			distances_sqd = distances * distances
			c_ss = sum(distances_sqd)/(len(distances_sqd)-1)
			clusters_ss.append(c_ss)

		wcss = sum(clusters_ss)/len(clusters_ss)
		
		self.wcss = wcss

	def compute_bcss(self):
		all_bc_ss =[]

		c_nix = self.rowix(self.unique_centroids[0])
		cluster_vectors = self.neurons[c_nix,]

		for cluster_ix in range(1, len(self.unique_centroids)):
			c_nix = self.rowix(self.unique_centroids[cluster_ix])
			c_vector = self.neurons[c_nix,]
			cluster_vectors = np.vstack((cluster_vectors,c_vector))

		for cluster_ix in range(1, len(self.unique_centroids)):
			c_nix = self.rowix(self.unique_centroids[cluster_ix])
			c_vector = self.neurons[c_nix,]
			compute_vectors = np.vstack((c_vector,cluster_vectors))
			bc_distances = euclidean_distances(compute_vectors,compute_vectors)[0]
			bc_distances_sqd = bc_distances * bc_distances
			bc_ss = sum(bc_distances_sqd)/(len(bc_distances_sqd)-2) # cluster.ix appears twice
			all_bc_ss.append(bc_ss)

		bcss = sum(all_bc_ss)/len(all_bc_ss)
		self.bcss = bcss

	def accuracy(self, sample, data_ix):
		""" accuracy -- the topographic accuracy of a single sample is 1 is the best matching unit
		             	and the second best matching unit are are neighbors otherwise it is 0
		"""

		o = self.best_match(sample, full=True)
		best_ix = o[0]
		second_best_ix = o[1]

		# sanity check
		coord = self.coordinate(best_ix)
		coord_x = coord['x']
		coord_y = coord['y']

		map_ix = self.fitted_obs[data_ix-1]  # self.visual[data_ix-1]
		coord = self.coordinate(map_ix)
		map_x = coord['x']
		map_y = coord['y']

		if (coord_x != map_x or coord_y != map_y or best_ix != map_ix):
			print("Error: best_ix: ", best_ix, " map_ix: ", map_ix, "\n")

		# determine if the best and second best are neighbors on the map
		best_xy = self.coordinate(best_ix)
		second_best_xy = self.coordinate(second_best_ix)
		diff_map = np.array(list(best_xy.values())) - np.array(list(second_best_xy.values()))
		diff_map_sq = diff_map * diff_map
		sum_map = np.sum(diff_map_sq)
		dist_map = np.sqrt(sum_map)

		# it is a neighbor if the distance on the map
		# between the bmu and 2bmu is less than 2,   should be 1 or 1.414
		if dist_map < 2:
			return 1
		else:
			return 0

	def significance(self, graphics=True, feature_labels=False):

		data_df = self.data
		nfeatures = data_df.shape[1]

	    # Compute the variance of each feature on the map
		var_v = [randint(1, 1) for _ in range(nfeatures)]

		for i in range(nfeatures):
			var_v[i] = np.var(np.array(data_df)[:, i])

	    # we use the variance of a feature as likelihood of
	    # being an important feature, compute the Bayesian
	    # probability of significance using uniform priors

		var_sum = np.sum(var_v)
		prob_v = var_v/var_sum

	    # plot the significance
		if graphics:
			y = max(prob_v)

			plt.axis([0, nfeatures+1, 0, y])

			x = np.arange(1, nfeatures+1)
			tag = list(data_df)

			plt.xticks(x, tag)
			plt.yticks = np.linspace(0, y, 5)

			i = 1
			for xc in prob_v:
				plt.axvline(x=i, ymin=0, ymax=xc)
				i += 1

			plt.xlabel('Features')
			plt.ylabel('Significance')
			plt.show()
		else:
			return prob_v

	def coordinate(self, rowix):
		x = (rowix) % self.xdim
		y = (rowix) // self.xdim
		return {'x':x,'y':y}

	def rowix(self, coord):
		
		rix = coord['x'] + coord['y']*self.xdim
		return rix

	def starburst(self):

		heat = self.heat
		x = self.xdim
		y = self.ydim

		if (x <= 1 or y <= 1):
			sys.exit("plot_heat: map dimensions too small")

		heat_tmp = np.squeeze(np.asarray(heat)).flatten()   # Convert 2D Array to 1D
		tmp = pd.cut(heat_tmp, bins=100, labels=False)
		tmp = np.reshape(tmp, (-1, y))						# Convert 1D Array to 2D

		tmp_1 = np.array(np.matrix.transpose(tmp))
		fig, ax = plt.subplots()
		ax.pcolor(tmp_1, cmap=plt.cm.YlOrRd)
		ax.set_xticks(np.arange(x)+0.5, minor=False)
		ax.set_yticks(np.arange(y)+0.5, minor=False)
		plt.xlabel("x")
		plt.ylabel("y")
		ax.set_xticklabels(np.arange(x), minor=False)
		ax.set_yticklabels(np.arange(y), minor=False)
		ax.xaxis.set_tick_params(labeltop='on')
		ax.yaxis.set_tick_params(labelright='on')

		centroids = self.centroids
		for ix in range(x):
			for iy in range(y):
				cx = centroids[ix][iy]['x']
				cy = centroids[ix][iy]['y']
				plt.plot([ix+0.5, cx+0.5],
		                       [iy+0.5, cy+0.5],
		                       color='lightgrey',
		                       linestyle='-',
		                       linewidth=1.0)
		        
		centroid_labels = self.centroid_labels

		for ix in range(x):
			for iy in range(y):
				lab = centroid_labels[ix][iy]
				if lab:
					plt.text(ix+0.5, iy+0.5, lab)


		plt.show()

	def summary(self):
		value = []

		header = [  "xdim",
        		    "ydim",
            		"alpha",
            		"train",
            		"normalize",
            		"seed",
            		"instances"]

		if self.normalize:
			v_normalize = True
		else:
			v_normalize = False

		if self.seed:
			v_seed = m.seed
		else:
			v_seed = None
    
		v = pd.DataFrame([[self.xdim,
        		        self.ydim,
                		self.alpha,
		                self.train,
                		v_normalize,
                		v_seed,
                		self.data.shape[0]]], columns=header)

		self.training_parameters = v

		print("Training Parameters:\n\n",v)
    
		header = ["convergence","separation","clusters"]
		v = pd.DataFrame([[ self.convergence, 
        		            1.0 - self.wcss/self.bcss,
                		    len(self.unique_centroids)]],columns=header)

		self.quality_assessments = v

		print("\n\n Quality Assessments:\n\n",v)
