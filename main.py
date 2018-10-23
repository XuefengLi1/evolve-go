import tf_util as U
import gym
import random
from policies import Policy,GoPolicy
from es import *
from mpi4py import MPI
import argparse, sys, os, pdb

# turn off tensorflow's warnings
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
    dict(index=11,game="Humanoid-v2",continuous_a=[True,0.4], ep_max_step=20000, eval_threshold=800,atari=False),
    dict(index=12,game="Swimmer-v1",continuous_a=[True,1], ep_max_step=20000, eval_threshold=800,atari=False),
    dict(index=13,game="Walker2d-v2",continuous_a=[True,1], ep_max_step=20000, eval_threshold=800,atari=False),
    dict(index=14,game="Go9x9-v0",continuous_a=[False], ep_max_step=200, eval_threshold=-120,atari=False),
]



dic = {}


def main(args):

    # Create Gym env
    env = gym.make(CONFIG[args.game]['game']).unwrapped

    # Set the continuity of the env
    env.continuous = CONFIG[args.game]['continuous_a'][0]

    summary = True if rank == 0 and args.summary else False

    # Create the policy(network)
    policy = GoPolicy(env, scope='mutant_net', mean_pol=True,summary=summary) if args.game == 14 else Policy(env, scope='mutant_net', summary=summary)

    if summary:
        monitor = U.Summarizer(np.array(1.,dtype=np.float32),policy)
        monitor2 = U.Summarizer(np.array(1., dtype=np.float32), policy)

    # Get the number of variables
    dim = int(policy.dimension)

    es = OpenES(policy, dim,sigma_init=args.sig_init,learning_rate=args.lr,popsize=size,weight_decay=args.weight_decay, fitness_shaping= args.f_shaping)


    if args.load:es.load(args.load)

    # Create the optimizers Adam/SGD with momentum
    optimizer = SGD(es,es.learning_rate)

    # Create buffers for receiving results from other processes
    results = np.empty(size, dtype=np.float32)
    mirrored_results = np.empty(size, dtype=np.float32)
    seeds = np.empty(size, dtype='i')

    # running = 0
    repeat = 0

    random.seed(os.getpid())

    for i in range(100000):

        # Random generate new seed for each iteration
        noise_seed = np.array(random.randint(0, 2 ** 16 -1),dtype='i')

        sed = np.asscalar(noise_seed)

        if int(sed) in dic:
            if rank ==0:
                print(sed)
            repeat += 1
        else:
            dic[sed] = 1

        # Generate samples with the random seed
        sample = es.generate(noise_seed)

        # Rollout
        if args.game == 14:policy.mean_pol.setVariables(es.mu)

        summary = True if rank == 0 and i % 10 == 0 and args.summary else False

        result, t = policy.rollout(sample[0],summary=summary)

        mirrored_result, mirrored_t = policy.rollout(sample[1])

        # Send and receive all the results and seeds from/to other processes
        comm.Allgather([result, MPI.FLOAT],[results, MPI.FLOAT])
        comm.Allgather([mirrored_result, MPI.FLOAT],[mirrored_results, MPI.FLOAT])
        comm.Allgather([noise_seed, MPI.INT],[seeds, MPI.INT])

        # Concatenate mirrored sampling results
        combined_results = np.concatenate([results, mirrored_results])

        # Calculate the updating gradient
        gradient = es.gradient_cal((combined_results, seeds))

        # Update with optimizer
        step = optimizer.update(gradient - es.weight_decay*es.mu)


        if args.save and rank == 0 and i % 1000 == 0:
            es.save()
        #
        if rank == 0 and args.render:
            policy.rollout(es.mu,render=args.render,summary=args.summary)

        if rank == 0 and i % 10 == 0:
            # result, t = policy.rollout(es.mu, summary=args.summary)
            # print("iteration %d       reward of mean: %d        mean_reward: %d" %(i,np.asscalar(result),np.asscalar(combined_results.mean())))
            max_r = combined_results.max()
            mean_r = combined_results.mean()
            print(result)
            if summary:
                monitor.add_summary(np.array(max_r))
                monitor2.add_summary(np.array(mean_r))
            print("iteration %d       reward of max: %d        mean_reward: %d" %(i,np.asscalar(combined_results.max()),np.asscalar(combined_results.mean())))

            sys.stdout.flush()


    print("number of repeated seed: ",repeat)

    env.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evo Strategiser')
    parser.add_argument('--game', default=0, type=int, help='index of game')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--pop_size', default=8, type=int, help='population_size')
    parser.add_argument('--sig_init', default=0.02, type=float, help='initial sigma')
    parser.add_argument('--weight_decay', default=0.005, type=float, help='weight decay')

    parser.add_argument('--load', default=None,type=str, help='Loaded model path')

    parser.add_argument('--f_shaping', default=False, action="store_true", help='fitness shaping')
    parser.add_argument('--save', default=False, action="store_true", help='save model')
    parser.add_argument('--render', default=False, action="store_true", help='Whether the first worker (worker_index==0) should render the environment')
    parser.add_argument('--debug', default=False, action="store_true", help='Whether to use the debug log level')
    parser.add_argument('--summary', default=False, action="store_true", help='Whether to use tensorflow summary')

    args = parser.parse_args()

    main(args)
