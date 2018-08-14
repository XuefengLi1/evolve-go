import tf_util as U
import gym
import numpy as np
from random import randint
from policies import Policy
from es import *
import pdb
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

GAME = 0
CONFIG = [
    dict(index=0,game="CartPole-v0", continuous_a=[False], ep_max_step=700, eval_threshold=199,atari=False),
    dict(index=1,game="MountainCar-v0",continuous_a=[False], ep_max_step=200, eval_threshold=-120,atari=False),
    dict(index=2,game="Pendulum-v0",continuous_a=[True, 2.], ep_max_step=200, eval_threshold=-180,atari=False),
    dict(index=3,game="BipedalWalker-v2",continuous_a=[True, 1.], ep_max_step=500, eval_threshold=100,atari=False),
    dict(index=4,game="Acrobot-v1",continuous_a=[False], ep_max_step=200, eval_threshold=0,atari=False),
    dict(index=5,game="Breakout-ram-v0",continuous_a=[False], ep_max_step=2000, eval_threshold=100,atari=False),
    dict(index=6,game="LunarLander-v2",continuous_a=[False], ep_max_step=10000, eval_threshold=190,atari=False),
    dict(index=7,game="BreakoutNoFrameskip-v4",continuous_a=[False], ep_max_step=2000, eval_threshold=100,atari=True),
    dict(index=8,game="PongNoFrameskip-v0",continuous_a=[False], ep_max_step=20000, eval_threshold=21,atari=True),
    dict(index=9,game="InvertedPendulum-v1",continuous_a=[True,1000], ep_max_step=20000, eval_threshold=800,atari=False),
    dict(index=10,game="InvertedDoublePendulum-v1",continuous_a=[True,1000], ep_max_step=20000, eval_threshold=800,atari=False),
    dict(index=11,game="Humanoid-v1",continuous_a=[True,0.4], ep_max_step=20000, eval_threshold=800,atari=False),
    dict(index=12,game="Swimmer-v1",continuous_a=[True,1], ep_max_step=20000, eval_threshold=800,atari=False),
    dict(index=13,game="Walker2d-v1",continuous_a=[True,1], ep_max_step=20000, eval_threshold=800,atari=False)
][GAME]






def main():

    scope = 'net'

    env = gym.make(CONFIG['game'])

    policy = Policy(env, scope)

    dim = int(policy.dimension)

    es = OpenES(policy, dim,sigma_init=0.02,learning_rate=0.01,popsize=4,antithetic=False,weight_decay=0.005)

    optimizer = SGD(es,es.learning_rate)

    results = np.empty(size, dtype=np.float32)

    seeds = np.empty(size, dtype='i')

    for i in range(2000):

        noise_seed = np.array(randint(0, 2 ** 16 -1),dtype='i')

        sample = es.generate(noise_seed)

        result, t = policy.rollout(sample[0])

        comm.Allgather([result, MPI.INT],[results, MPI.INT])
        comm.Allgather([noise_seed, MPI.INT],[seeds, MPI.INT])

        gradient = es.gradient_cal((results, seeds))

        step = optimizer.update(gradient)

        if results.mean() > 190:
            if rank == 0:
                result, t = policy.rollout(es.mu, render=True)
            break


    env.close()


if __name__ == '__main__':
    main()
    # print("--- %s seconds ---" % (time.time() - start_time))
