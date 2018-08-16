import numpy as np
import tensorflow as tf
import tf_util as U

class Policy:
	def __init__(self, env, scope):

		self.num_actions = env.action_space.n
		self.scope = scope
		self.env = env
		self.build_model()

	def build_model(self):
		with tf.device("/cpu:0"):
			with tf.variable_scope(self.scope):

				self.build_graph()

			self.all_variables = tf.trainable_variables(scope=self.scope)

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

	@property	
	def dimension(self):
		return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables(scope=self.scope)])

	def intprod(self,x):
		return int(np.prod(x))

	def save_model(self, iternum):
		# path = '/short/mf16/xl1369/dense/saved-models/checkpoints'
		path = '../saved-models/checkpoints'
		self.saver.save(self.sess, path + str(iternum))

	# Summary operations for tensorboard
	def summary(self):
		self.merged = tf.summary.merge_all()
		self.writer = tf.summary.FileWriter('logs/',self.sess.graph)

	def build_graph(self):
		self.observation = tf.placeholder(tf.float32, [None] + list(self.env.observation_space.shape), name='inputs')
		# out = tf.layers.dense(self.observation, 10, use_bias=True, activation=tf.nn.tanh)
		out = U.dense(self.observation, 10, 'layer1', weight_init=None, bias=False)
		tf.summary.histogram('layer1_activations', out)
		out = tf.nn.tanh(out)
		# out = tf.layers.dense(out, 10, use_bias=True, activation=tf.nn.tanh)
		out = U.dense(out, 10, 'layer2', weight_init=None, bias=False)
		tf.summary.histogram('layer2_activations', out)
		out = tf.nn.tanh(out)
		self.actions = tf.layers.dense(out, self.num_actions, use_bias=True, activation=None,name='outputs')

	def act(self, obv):

		actions = self.sess.run(self.actions, feed_dict={self.observation:obv})
		actions_pass = np.append(actions[0],0)
		result = np.argmax(actions[0])

		return result

	def act_summary(self, obv):

		actions, summary = self.sess.run([self.actions,self.merged], feed_dict={self.observation:obv})
		self.writer.add_summary(summary)

		actions_pass = np.append(actions[0],0)
		result = np.argmax(actions[0])

		return result 

	def rollout(self, sample, render=False, timestep_limit=None):

		env = self.env
		
		self.setVariables(sample)

		env_timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
		timestep_limit = env_timestep_limit if timestep_limit is None else min(timestep_limit, env_timestep_limit)
		rews = []
		t = 0

		ob = env.reset()
		for _ in range(timestep_limit):
			ac = self.act([ob])

			ob, rew, done, _ = env.step(ac)
			rews.append(rew)
			t += 1
			if render:
				env.render()
			if done:
				break
		rews = np.array(rews, dtype=np.float32)

		return np.sum(rews,dtype=np.float32), t

	def rollout_summary(self, sample, render=False, timestep_limit=None):

		env = self.env
		
		self.setVariables(sample)

		env_timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
		timestep_limit = env_timestep_limit if timestep_limit is None else min(timestep_limit, env_timestep_limit)
		rews = []
		t = 0

		ob = env.reset()
		for _ in range(timestep_limit):
			ac = self.act_summary([ob])

			ob, rew, done, _ = env.step(ac)
			rews.append(rew)
			t += 1
			if render:
				env.render()
			if done:
				break
		rews = np.array(rews, dtype=np.float32)

		return np.sum(rews,dtype=np.float32), t

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

	def __init__(self, env, scope):
		super(GoPolicy,self).__init__(env, scope)
		self.num_actions = env.action_space.n - 1
		self.build_model()

	def build_graph(self):

		def pad(board):

			board = tf.pad(board,[[0,0],[0,0],[3,3],[3,3]],constant_values=0.0)
			edges = tf.constant(0,dtype=tf.float32,shape=[1,1,9,9])
			edges = tf.pad(edges,[[0,0],[0,0],[3,3],[3,3]],constant_values=1.0)
			board = tf.concat([board,edges],1)

			return board

		self.observation = tf.placeholder(tf.float32, [None] + list(self.env.observation_space.shape), name='inputs')

		board = pad(self.observation)

		# Channel first to channel last as conv2d only support channel last
		board = tf.transpose(board,[0,2,3,1])

		conv1 = tf.layers.conv2d(board, filters=16, kernel_size=5, strides=1, use_bias=True, activation=tf.nn.relu, padding="valid")

		conv2 = tf.layers.conv2d(conv1, filters=16, kernel_size=5, strides=1,use_bias=True, activation=tf.nn.relu, padding="valid")

		conv3 = tf.layers.conv2d(conv2, filters=16, kernel_size=5, strides=1,use_bias=True, activation=tf.nn.relu, padding="valid")

		conv4 = tf.layers.conv2d(conv3, filters=16, kernel_size=5, strides=1,use_bias=True, activation=tf.nn.relu, padding="valid")

		out = tf.layers.conv2d(conv4, filters=1, kernel_size=1, strides=1, use_bias=True, activation=None, padding="same")

		output = tf.layers.flatten(out)
		
		self.actions = tf.identity(output, 'outputs')

	def select_action(self, obv):

		actions = self.sess.run(self.actions, feed_dict={self.observation:obv})

		actions_pass = np.append(actions[0],0)

		result = np.argsort(actions_pass)

		return result
