import numpy as np
import pdb
from optimizers import *

STAT_RANGE = 10

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
                 fitness_shaping=False,             # whether to use antithetic sampling
                 weight_decay=0.005):            # weight decay coefficient

        self.num_params = num_params
        self.sigma = sigma_init * np.ones(self.num_params)
        self.learning_rate = learning_rate

        self.popsize = popsize


        self.mu = policy.get_trainable_flat()

        self.mu *= 0

        self.weight_decay = weight_decay
        self.utilities = shaped_fit(popsize)
        self.fitness_shaping = fitness_shaping

        self.fstat_sum = 0.
        self.fstat_sumsq = 0.
        self.fstat_count = 0.

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

        assert(len(results) == self.popsize*2), "Inconsistent reward_table size reported."
        assert(len(seeds) == self.popsize), "Inconsistent reward_table size reported."


        #if self.fitness_shaping:



        size = len(results)
        assert(size % 2 == 0), "Inconsistent result size for mirrored sampling reported."



        gradient = np.zeros(self.num_params)

        # Reconstruct epsilon
        noise = []
        for seed in seeds:
            np.random.seed(seed)
            noise.append(np.random.randn(self.num_params))

        # if not self.fitness_shaping:
        reward = compute_centered_ranks(results)
        idx = np.argsort(reward)[::-1]
        for ui, i in enumerate(idx):
            seed_index = i - (size/2) if i >= size/2 else i
            seed_index = int(seed_index.item())

            sign1 = sign(i, size/2)
            gradient += self.utilities[ui] * noise[seed_index] * sign1
        # else:
        #
        #     f_mean, f_stdv = self.fitness_stat(results)
        #
        #     reward = (results - f_mean) / f_stdv
        #
        #     assert (True not in np.isnan(reward), "Nan in reward")
        #
        #     if f_stdv == 0.0: return 0
        #
        #     for ui, i in enumerate(reward):
        #         seed_index = ui - (size/2) if ui >= size/2 else ui
        #         seed_index = int(seed_index)
        #
        #         sign1 = sign(i, size/2)
        #         if i <= 0: continue
        #         gradient += i * noise[seed_index] * sign1
        # self.mu += self.learning_rate * change_mu


        gradient /= self.popsize*2
                     # *self.sigma)

        assert (not np.isnan(np.sum(gradient)), "Nan in gradient")
        return gradient

    # update_ratio = self.optimizer.update(-gradient)

    def save(self):
        np.savetxt('model/weights.out', self.mu, fmt='%.18e')

    def load(self, file):
        self.mu = np.loadtxt(file, np.float32)

    def fitness_stat(self,results):
        self.fstat_count += 1
        # if (self.fstat_count <= STAT_RANGE):
        #     self.fstat_sum = self.fstat_sum + results.mean()
        #     self.fstat_sumsq = self.fstat_sumsq + results.mean() * results.mean()
        # else:
        #     self.fstat_sum = (1.0 - 1.0 / STAT_RANGE) * self.fstat_sum + results.mean()
        #     self.fstat_sumsq = (1.0 - 1.0 / STAT_RANGE) * self.fstat_sumsq + results.mean() * results.mean()
        #
        # if (self.fstat_count <= STAT_RANGE):
        #     f_mean = self.fstat_sum / self.fstat_count
        #     f_var = self.fstat_sumsq / self.fstat_count - f_mean * f_mean
        # else:
        #     f_mean = self.fstat_sum / STAT_RANGE
        #     f_var = self.fstat_sumsq / STAT_RANGE - f_mean * f_mean

        #return f_mean,np.sqrt(f_var)

        if (self.fstat_count <= STAT_RANGE):
            self.fstat_sum = self.fstat_sum + results.mean()
            self.fstat_sumsq = self.fstat_sumsq + np.std(results)
        else:
            self.fstat_sum = (1.0 - 1.0 / STAT_RANGE) * self.fstat_sum + results.mean()
            self.fstat_sumsq = (1.0 - 1.0 / STAT_RANGE) * self.fstat_sumsq + np.std(results)

        if (self.fstat_count <= STAT_RANGE):
            f_mean = self.fstat_sum / self.fstat_count
            f_std = self.fstat_sumsq / self.fstat_count
        else:
            f_mean = self.fstat_sum / STAT_RANGE
            f_std = self.fstat_sumsq / STAT_RANGE

        return f_mean, f_std