import numpy as np
from optimizers import *
import pdb

def compute_ranks(x):
	"""
	Returns ranks in [0, len(x))
	Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
	(https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
	"""
	assert x.ndim == 1
	ranks = np.empty(len(x), dtype=int)
	ranks[x.argsort()] = np.arange(len(x))
	return ranks

def compute_centered_ranks(x):
	"""
	https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
	"""
	y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
	y /= (x.size - 1)
	y -= .5
	return y

def compute_weight_decay(weight_decay, model_param_list):
	model_param_grid = np.array(model_param_list)
	return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)


def shaped_fit(size):
	base = size * 2  # *2 for mirrored sampling
	rank = np.arange(1, base + 1)
	util_ = np.maximum(0, np.log(base / 2 + 1) - np.log(rank))
	utility = util_ / util_.sum() - 1 / base
	return utility

def sign(idx, size): return -1. if idx >=size else 1.

class OpenES:
	''' Basic Version of OpenAI Evolution Strategies.'''
	def __init__(self, policy, num_params,             # number of model parameters
							 sigma_init=0.05,               # initial standard deviation
							 learning_rate=0.01,           # learning rate for standard deviation
							 popsize=1,                  # population size
							 antithetic=False,             # whether to use antithetic sampling
							 weight_decay=0.005):            # weight decay coefficient

		self.num_params = num_params
		self.sigma = sigma_init * np.ones(self.num_params)
		self.learning_rate = learning_rate

		self.popsize = popsize
		# self.antithetic = antithetic

		# if self.antithetic:
		#   assert (self.popsize % 2 == 0), "Population size must be even"
		#   self.half_popsize = int(self.popsize / 2)

		# self.mu = np.random.randn(self.num_params) * 0.1q

		self.mu = np.random.randn(self.num_params) * 0.01

		self.weight_decay = weight_decay
		self.utilities = shaped_fit(popsize)

		# choose optimizer
		# self.optimizer = Adam(policy,learning_rate)


	def generate(self, seed):
		'''returns a list of parameters'''
		# antithetic sampling

		np.random.seed(seed)

		self.epsilon = np.random.randn(self.num_params)
		# print(self.epsilon)

		# if self.antithetic:
		self.epsilon = np.array([self.epsilon, - self.epsilon])
		# pdb.set_trace()

		self.samples = self.mu + self.epsilon * self.sigma

		return self.samples

	def gradient_cal(self, info):
		# input must be a numpy float array
		results, seeds = info

		# results = np.array([ 9., 10., 10., 11.,  9., 10., 10.,  8.],dtype=np.float32)
		#
		# seeds = np.array([14598, 29149, 41506, 24126], dtype='i')

		assert(len(results) == self.popsize*2), "Inconsistent reward_table size reported."
		assert(len(seeds) == self.popsize), "Inconsistent reward_table size reported."
		
		reward = compute_centered_ranks(results)

		size = len(results)
		assert(size % 2 == 0), "Inconsistent result size for mirrored sampling reported."

		idx = np.argsort(reward)[::-1]

		gradient = np.zeros(self.num_params)

		# Reconstruct epsilon
		noise = []
		for seed in seeds:
			np.random.seed(seed)
			noise.append(np.random.randn(self.num_params))

		for ui, i in enumerate(idx):
				seed_index = i - (size/2) if i >= size/2 else i
				seed_index = int(seed_index.item())

				sign1 = sign(i, size/2)
				gradient += self.utilities[ui] * noise[seed_index] * sign1

		#self.mu += self.learning_rate * change_mu
		gradient /= (self.popsize*2*self.sigma)

		return gradient

		# update_ratio = self.optimizer.update(-gradient)


