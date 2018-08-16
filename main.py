import tf_util as U
import gym
import numpy as np
from random import randint
from policies import Policy
from es import *
import pdb
from mpi4py import MPI
import argparse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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
]






def main(args):

    scope = 'net'
    env = gym.make(CONFIG[args.game]['game'])

    policy = Policy(env, scope)

    if rank == 0: policy.summary()

    dim = int(policy.dimension)

    es = OpenES(policy, dim,sigma_init=args.sig_init,
        learning_rate=args.lr,popsize=args.pop_size,antithetic=args.antithetic,weight_decay=args.weight_decay)

    optimizer = SGD(es,es.learning_rate)

    results = np.empty(size, dtype=np.float32)

    seeds = np.empty(size, dtype='i')

    for i in range(1000):

        noise_seed = np.array(randint(0, 2 ** 16 -1),dtype='i')

        sample = es.generate(noise_seed)

        if rank==0 and i%10==0: 
            result, t = policy.rollout_summary(sample[0])
        else:
            result, t = policy.rollout(sample[0])



        comm.Allgather([result, MPI.INT],[results, MPI.INT])
        comm.Allgather([noise_seed, MPI.INT],[seeds, MPI.INT])

        gradient = es.gradient_cal((results, seeds))

        step = optimizer.update(gradient - es.weight_decay * es.mu)

        # if results.max() >= 199:
        #     if rank == 0:
        #         print(optimizer.t)
        #         result, t = policy.rollout(es.mu, render=True)
        #     break


    env.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evo Strategiser')
    parser.add_argument('--game', default=0, type=int, help='index of game')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--pop_size', default=4, type=int, help='population_size')
    parser.add_argument('--antithetic', default=False, action="store_true", help='mirrored sampling')
    parser.add_argument('--sig_init', default=0.02, type=float, help='initial sigma')
    parser.add_argument('--weight_decay', default=0.005, type=float, help='weight decay')

    parser.add_argument('--render', default=False, action="store_true", help='Whether the first worker (worker_index==0) should render the environment')
    parser.add_argument('--debug', default=False, action="store_true", help='Whether to use the debug log level')

    args = parser.parse_args()

    main(args)
    # print("--- %s seconds ---" % (time.time() - start_time))
