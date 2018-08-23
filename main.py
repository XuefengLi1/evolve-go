import tf_util as U
import gym
from random import randint
from policies import Policy
from es import *
from mpi4py import MPI
import argparse, sys, os

# tensorflow's warnings are too annoying
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
    dict(index=13,game="Walker2d-v2",continuous_a=[True,1], ep_max_step=20000, eval_threshold=800,atari=False)
]



dic = {}


def main(args):

    # Specify the name scope for tensorflow graph
    scope = 'net'

    # Create Gym env
    env = gym.make(CONFIG[args.game]['game']).unwrapped

    # Set the continuity of the env
    env.continuous = CONFIG[args.game]['continuous_a'][0]

    # Create the policy(network)
    policy = Policy(env, scope)

    # Get the number of variables
    dim = int(policy.dimension)

    #
    es = OpenES(policy, dim,sigma_init=args.sig_init,learning_rate=args.lr,popsize=size,antithetic=args.antithetic,weight_decay=args.weight_decay)

    # Create the optimizers Adam/SGD with momentum
    optimizer = Adam(es,es.learning_rate)

    # Create buffers for receiving results from other processes
    results = np.empty(size, dtype=np.float32)
    mirrored_results = np.empty(size, dtype=np.float32)
    seeds = np.empty(size, dtype='i')

    # Creat summary histogram for statistics

    if rank == 0: summarizer = U.Summarizer(es.mu, policy)
    if rank == 0: policy.summary()


    # running = 0
    repeat = 0
    for i in range(5000):

        # Random generate new seed for each iteration
        noise_seed = np.array(randint(0, 2 ** 16 -1),dtype='i')

        # sed = np.asscalar(noise_seed)
        #
        # if int(sed) in dic:
        #     repeat += 1
        # else:
        #     dic[sed] = 1

        # Generate samples with the random seed
        sample = es.generate(noise_seed)

        # Rollout
        result, t = policy.rollout(sample[0])
        mirrored_result, mirroed_t = policy.rollout(sample[1])

        # Send and receive all the results and seeds from/to other processes
        comm.Allgather([result, MPI.INT],[results, MPI.INT])
        comm.Allgather([mirrored_result, MPI.INT],[mirrored_results, MPI.INT])
        comm.Allgather([noise_seed, MPI.INT],[seeds, MPI.INT])

        # Conatenate mirrored sampling results
        combined_results = np.concatenate([results, mirrored_results])

        # Calculate the updating gradient
        gradient = es.gradient_cal((combined_results, seeds))

        # Update with optimizer
        step = optimizer.update(gradient - es.weight_decay*es.mu)

        if rank == 0:
            # result, t = policy.rollout(es.mu)
            # print("iteration %d       reward of mean: %d        mean_reward: %d" %(i,np.asscalar(result),np.asscalar(combined_results.mean())))
            print("iteration %d       reward of max: %d        mean_reward: %d" %(i,np.asscalar(combined_results.max()),np.asscalar(combined_results.mean())))

            sys.stdout.flush()


    print("number of repeated seed: ",repeat)

    env.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evo Strategiser')
    parser.add_argument('--game', default=0, type=int, help='index of game')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--pop_size', default=8, type=int, help='population_size')
    parser.add_argument('--antithetic', default=False, action="store_true", help='mirrored sampling')
    parser.add_argument('--sig_init', default=0.02, type=float, help='initial sigma')
    parser.add_argument('--weight_decay', default=0.005, type=float, help='weight decay')

    parser.add_argument('--render', default=False, action="store_true", help='Whether the first worker (worker_index==0) should render the environment')
    parser.add_argument('--debug', default=False, action="store_true", help='Whether to use the debug log level')
    parser.add_argument('--summary', default=False, action="store_true", help='Whether to use tensorflow summary')

    args = parser.parse_args()

    main(args)