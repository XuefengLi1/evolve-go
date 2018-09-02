import numpy as np
import tensorflow as tf
import tf_util as U
import random
import re
import pdb
class Policy:
    def __init__(self, env, scope, summary=False):

        self.scope = scope
        self.env = env
        self.num_actions = env.action_space.n if not env.continuous else env.action_space.shape[0]
        self.summary = summary
        self.build_model()

        if summary:
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter('logs/'+env.spec.id, self.sess.graph)

    def build_model(self):

        with tf.device("/cpu:0"):
            with tf.variable_scope(self.scope):

                self.build_graph()

            self.all_variables = tf.trainable_variables(scope=self.scope)

            self._getflat = U.GetFlat(self.all_variables)

            self.placeholders = [tf.placeholder(v.value().dtype, v.get_shape().as_list()) for v in self.all_variables]

            self.sess = tf.InteractiveSession() if tf.get_default_session() == None else tf.get_default_session()

            self.sess.run(tf.global_variables_initializer())

            self.vars,self.assigns = self.init_attr()


            self.saver = tf.train.Saver()




    def init_attr(self):
        vars = [(v, self.intprod(v.get_shape()),v.get_shape()) for v in self.all_variables]
        assigns = []
        for i in range(len(self.all_variables)):
            assigns.append(tf.assign(self.all_variables[i],self.placeholders[i]))
        return vars,assigns

    def setVariables(self, theta):
        cur = 0
        data = []
        for (var, size, shape) in self.vars:
            end = cur + size
            subarray = np.reshape(theta[cur:end], shape)
            data.append(subarray)
            cur = end

        feed_dict = dict(zip(self.placeholders, data))
        self.sess.run(self.assigns, feed_dict=feed_dict)

        return data

    def get_trainable_flat(self):
        return self._getflat()

    @property
    def dimension(self):
        return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables(scope=self.scope)])

    def intprod(self,x):
        return int(np.prod(x))

    def save_model(self, iternum):
        # path = '/short/mf16/xl1369/dense/saved-models/checkpoints'
        path = '../saved-models/checkpoints'
        self.saver.save(self.sess, path + str(iternum))

    def build_graph(self):
        self.observation = tf.placeholder(tf.float32, [None] + list(self.env.observation_space.shape), name='inputs')
        # out = self.observation

        out = U.dense(self.observation, 36, 'layer1', weight_init=tf.variance_scaling_initializer(), bias=True,activation=tf.nn.tanh, summary=self.summary)
        out = U.dense(out, 36, 'layer2', weight_init=tf.variance_scaling_initializer(), bias=True, activation=tf.nn.tanh, summary=self.summary)
        activation = tf.nn.tanh if self.env.continuous else None

        self.actions = U.dense(out, self.num_actions, 'output', weight_init=tf.variance_scaling_initializer(), bias=True, activation=activation, summary=self.summary)

        # self.actions = tf.layers.dense(out, self.num_actions, use_bias=True, activation=activation,name='outputs')

    def act(self, obv, summary=False):

        if summary:
            actions, summary = self.sess.run([self.actions, self.merged], feed_dict={self.observation: obv})
            self.writer.add_summary(summary)
        else:
            actions = self.sess.run(self.actions, feed_dict={self.observation: obv})

        # if self.env.continuous: scale = (self.env.action_space.high - self.env.action_space.low)/2

        result = actions[0] if self.env.continuous else np.argmax(actions[0])

        return result

    def rollout(self, sample, render=False, timestep_limit=None, summary=False):

        env = self.env

        self.setVariables(sample)

        env_timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
        timestep_limit = env_timestep_limit if timestep_limit is None else min(timestep_limit, env_timestep_limit)
        rews = []
        t = 0

        ob = env.reset()
        for _ in range(timestep_limit):
            ac = self.act([ob],summary)

            ob, rew, done, _ = env.step(ac)
            rews.append(rew)
            t += 1
            if render:
                env.render()
            if done:
                break
        rews = np.array(rews, dtype='i')

        return np.sum(rews,dtype='i'), t

    # def virtualBN(tensor,file, size):

    #     mean, vars = getStats(file)
    #     scale = tf.Variable(tf.ones([size]))
    #     offset = tf.Variable(tf.zeros([size]))
    #     normed_tensor = tf.nn.batch_normalization(tensor,mean,vars,offset,scale,0.001)
    #     return normed_tensor

    # def getStats(file):
    #     with open(file, 'rb')  as f:
    #         ref = np.load(f).astype(dtype=np.float32)
    #         mean = np.mean(ref,axis=0)
    #         vars_ = np.var(ref,axis=0)

    #     return mean,vars_

class GoPolicy(Policy):

    def __init__(self, env, scope, mean_pol=False,summary=False):
        self.obs_space = [1, 3, 9, 9]
        self.num_actions = env.action_space.n - 1
        super(GoPolicy,self).__init__(env, scope,summary=summary)
        if mean_pol: self.mean_pol = GoPolicy(env, scope='mean_net', summary=False)

    def build_graph(self):

        def pad(board):

            board = tf.pad(board,[[0,0],[0,0],[8,8],[8,8]],constant_values=0.0)
            edges = tf.constant(0,dtype=tf.float32,shape=[1,1,9,9])
            edges = tf.pad(edges,[[0,0],[0,0],[8,8],[8,8]],constant_values=1.0)
            board = tf.concat([board,edges],1)

            return board

        # activation function
        activation = tf.nn.selu

        self.observation = tf.placeholder(tf.float32, self.obs_space, name='inputs')

        board = pad(self.observation)


        # Channel first to channel last as conv2d only support channel last
        board = tf.transpose(board,[0,2,3,1])

        conv1 = U.conv(board, name='conv1',filters=16, kernel_size=7, strides=1, use_bias=True, activation=activation, padding="valid",summary=self.summary)

        conv2 = U.conv(conv1,name='conv2',filters=16, kernel_size=5, strides=1,use_bias=True, activation=activation, padding="valid",summary=self.summary)

        conv3 = U.conv(conv2, name='conv3', filters=16, kernel_size=5, strides=1,use_bias=True, activation=activation, padding="valid",summary=self.summary)

        conv4 = U.conv(conv3, name='conv4', filters=16, kernel_size=3, strides=1, use_bias=True, activation=activation,padding="valid",summary=self.summary)

        out = U.conv(conv4, name='out',filters=1, kernel_size=1, strides=1, use_bias=True, activation=None, padding="valid",summary=self.summary)

        output = tf.layers.flatten(out)

        #self.actions = tf.identity(output, 'outputs')
        self.actions = output

    def act(self, obv,summary=False):

        if summary and self.summary:
            actions, summary = self.sess.run([self.actions, self.merged], feed_dict={self.observation: obv})
            self.writer.add_summary(summary)
        else:
            actions = self.sess.run(self.actions, feed_dict={self.observation: obv})

        actions_pass = np.append(actions[0],0)

        result = np.argsort(actions_pass)

        return result

    def rollout(self, sample, render=False, timestep_limit=None, summary=False):
        # Swap first two channels and keep the empty position channel
        def swap_obv(obv):
            obv1 = obv[:2]
            obv1 = obv1[::-1]
            obv1 = np.append(obv1, [obv[-1]], axis=0)

            return obv1

        # Rescale observation with rescale size

        def rescaling(obv, size):

            obv = obv * size

            return obv

        # inner function
        def single_trial(env, act_fns, rand_move, render=False, random=False,summary=False):
            obv = env.reset()
            # random moves
            for move in rand_move:
                obv, _, _, _ = env.step([move])
            player = 0
            # rescale
            # rescale_size = 4
            # obv = rescaling(obv, rescaling)

            for i in range(162):
                if render:
                    env.render()

                # champ or random turn
                if player == 1:
                    if random:
                        actions = np.arange(82)
                        np.random.shuffle(actions)
                    else:

                        # swap the channel
                        actions = act_fns[player]([swap_obv(obv)],summary=summary)[::-1]

                # mutant turn
                else:
                    actions = act_fns[player]([obv],summary=summary)[::-1]

                done = False

                obv, reward, done, _ = env.step(actions)

                # switch
                player = (player + 1) % 2

                result = reward

                if done:
                    break

            balck_captures = re.findall(r'Captures B: ([0-9]*) W', str(env.state))[0]

            white_captures = re.findall(r'W: ([0-9]*)', str(env.state))[0]

            result = result + int(white_captures) - int(balck_captures)

            return result

        self.setVariables(sample)

        # generate 4 random moves
        rand_move = random.sample(range(0, 81), 4)

        # mutant = black, champ = white
        first_result = single_trial(self.env, [self.act, self.mean_pol.act], rand_move,summary=summary)

        # reverse
        second_result = single_trial(self.env, [self.mean_pol.act, self.act], rand_move,summary=summary)

        # evaluation current policy against random policy
        evaluation = single_trial(self.env,[self.mean_pol.act, self.act], rand_move,random=True,summary=summary)

        # reverse the sign as the result = white score - black score
        result = second_result - first_result

        # print('firts: ', first_result, ' second: ', second_result, ' final: ',result)
        return np.array(result,dtype='i'), -evaluation
