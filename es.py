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


class OpenES:
	''' Basic Version of OpenAI Evolution Strategies.'''
	def __init__(self, policy, num_params,             # number of model parameters
							 sigma_init=0.05,               # initial standard deviation
							 learning_rate=0.01,           # learning rate for standard deviation
							 popsize=1,                  # population size
							 antithetic=False,             # whether to use antithetic sampling
							 weight_decay=0.005):            # weight decay coefficient

		self.num_params = num_params
		self.sigma = sigma_init
		self.learning_rate = learning_rate

		self.popsize = popsize
		# self.antithetic = antithetic

		# if self.antithetic:
		#   assert (self.popsize % 2 == 0), "Population size must be even"
		#   self.half_popsize = int(self.popsize / 2)

		self.mu = np.zeros(self.num_params)


		self.weight_decay = weight_decay

		# choose optimizer
		# self.optimizer = Adam(policy,learning_rate)


	def generate(self, seed):
		'''returns a list of parameters'''
		# antithetic sampling

		np.random.seed(seed)

		self.epsilon = np.random.randn(self.num_params)
		# print(self.epsilon)
		# if self.antithetic:
		#   self.epsilon = np.concatenate([self.epsilon_half, - self.epsilon_half])

		self.solutions = self.mu.reshape(1, self.num_params) + self.epsilon * self.sigma

		return self.solutions

	def gradient_cal(self, info):
		# input must be a numpy float array
		results, seeds = info

		assert(len(results) == self.popsize), "Inconsistent reward_table size reported."
		assert(len(seeds) == self.popsize), "Inconsistent reward_table size reported."
		
		reward = compute_centered_ranks(results)
		# reward = results/200
		idx = np.argsort(reward)[::-1]

		gradient = np.zeros(self.num_params)
		for ui, i in enumerate(idx):
				np.random.seed(seeds[i])                # reconstruct noise using seed
				gradient += reward[ui] * np.random.randn(self.num_params)
						# change_mu = 1./(self.popsize*self.sigma)*np.dot(self.epsilon.T, reward)
		
		#self.mu += self.learning_rate * change_mu
		gradient /= (self.popsize*self.sigma)
		return -gradient

		# update_ratio = self.optimizer.update(-gradient)


