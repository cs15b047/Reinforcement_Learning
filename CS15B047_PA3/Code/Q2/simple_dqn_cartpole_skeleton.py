import gym
import random

import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
import pickle
import time

class DQN:
	
	REPLAY_MEMORY_SIZE = 100000			# number of tuples in experience replay  
	EPSILON = 1.0 						# epsilon of epsilon-greedy exploation
	EPSILON_DECAY = 0.95				# exponential decay multiplier for epsilon
	HIDDEN1_SIZE = 128 					# size of hidden layer 1
	HIDDEN2_SIZE = 128					# size of hidden layer 2
	EPISODES_NUM = 2000	 				# number of episodes to train on. Ideally shouldn't take longer than 2000
	MAX_STEPS = 200 					# maximum number of steps in an episode 
	LEARNING_RATE = 0.001 				# learning rate and other parameters for SGD/RMSProp/Adam
	#change-- originally 10
	MINIBATCH_SIZE = 20 				# size of minibatch sampled from the experience replay
	#MY CHANGE - originally 0.9
	DISCOUNT_FACTOR = 0.99 				# MDP's gamma
	TARGET_UPDATE_FREQ = 100 			# number of steps (not episodes) after which to update the target networks 
	LOG_DIR = './logs' 					# directory wherein logging takes place


	# Create and initialize the environment
	def __init__(self, env):
		self.env = gym.make(env)
		self.env.seed(int(time.time()))
		assert len(self.env.observation_space.shape) == 1
		self.input_size = self.env.observation_space.shape[0]		# In case of cartpole, 4 state features
		self.output_size = self.env.action_space.n					# In case of cartpole, 2 actions (right/left)
		np.random.seed(int(time.time()))
	
	# Create the Q-network
  	def initialize_network(self):
  	
  		self.replay_memory = {}
  		self.session = tf.Session()

  		# placeholder for the state-space input to the q-network
		self.x = tf.placeholder(tf.float64, [None, self.input_size])
		self.action = tf.placeholder(tf.int32, [None, self.output_size]) # Action which is selected is 1 , rest are 0
		self.target = tf.placeholder(tf.float64, [None, ])

		############################################################
		# Design your q-network here.
		# 
		# Add hidden layers and the output layer. For instance:
		# 
		# with tf.name_scope('output'):
		#	W_n = tf.Variable(
		# 			 tf.truncated_normal([self.HIDDEN_n-1_SIZE, self.output_size], 
		# 			 stddev=0.01), name='W_n')
		# 	b_n = tf.Variable(tf.zeros(self.output_size), name='b_n')
		# 	self.Q = tf.matmul(h_n-1, W_n) + b_n
		#
		#############################################################

		self.dense1 = tf.layers.dense(inputs=self.x,units=self.HIDDEN1_SIZE,activation=tf.nn.relu,use_bias=True)
		# self.dropout1 = tf.layers.dropout(inputs=self.dense1, rate=0.4)
		self.dense2 = tf.layers.dense(inputs=self.dense1,units=self.HIDDEN2_SIZE,activation=tf.nn.relu,use_bias=True)
		# self.dropout2 = tf.layers.dropout(inputs=self.dense2, rate=0.4)
		self.Q = tf.layers.dense(inputs=self.dense2,units=self.output_size,use_bias=True)
		
		############################################################
		# Next, compute the loss.
		#
		# First, compute the q-values. Note that you need to calculate these
		# for the actions in the (s,a,s',r) tuples from the experience replay's minibatch
		#
		# Next, compute the l2 loss between these estimated q-values and 
		# the target (which is computed using the frozen target network)
		#
		############################################################

		q_used_1 = tf.multiply(self.Q, tf.cast(self.action, tf.float64))
		q_used = tf.reduce_sum(q_used_1, axis=1)		

		self.loss = tf.reduce_sum(tf.square(self.target - q_used))
		
		############################################################
		# Finally, choose a gradient descent algorithm : SGD/RMSProp/Adam. 
		#
		# For instance:
		# optimizer = tf.train.GradientDescentOptimizer(self.LEARNING_RATE)
		# global_step = tf.Variable(0, name='global_step', trainable=False)
		# self.train_op = optimizer.minimize(self.loss, global_step=global_step)
		#
		############################################################
		optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
		global_step = tf.Variable(0, name='global_step', trainable=False)
		self.train_op = optimizer.minimize(self.loss, global_step=global_step)

		############################################################

	def train(self, episodes_num=EPISODES_NUM):
		
		saver = tf.train.Saver()

		# Initialize summary for TensorBoard
		summary_writer = tf.summary.FileWriter(self.LOG_DIR)
		summary = tf.Summary()
		# Alternatively, you could use animated real-time plots from matplotlib
		# (https://stackoverflow.com/a/24228275/3284912)
		
		# Initialize the TF session
		self.session = tf.Session()
		self.session.run(tf.global_variables_initializer())
		self.session2 = tf.Session()
		
		############################################################
		# Initialize other variables (like the replay memory)
		############################################################
		saver.save(self.session, "~/dqn")		
		saver.restore(self.session2,"~/dqn")
		total_steps = 0 
		oldest_experience = self.replay_memory.keys()[0]
		curr_experience_num = self.REPLAY_MEMORY_SIZE # Assuming replay buffer is already fully filled

		episode_window = []
		avg_epsiode_length = 0.0

		avg_episode_length_list = []
		ep_count = 0

		############################################################
		# Main training loop
		# 
		# In each episode, 
		#	pick the action for the given state,
		#	perform a 'step' in the environment to get the reward and next state,
		#	update the replay buffer,
		#	sample a random minibatch from the replay buffer,
		# 	perform Q-learning,
		#	update the target network, if required.
		#
		#
		#
		# You'll need to write code in various places in the following skeleton
		#
		############################################################

		for episode in range(episodes_num):

			# if episode % 50 == 0:
			# 	self.playPolicy()

			state = self.env.reset()

			############################################################
			# Episode-specific initializations go here.
			############################################################
			episode_length = 0
			episodic_reward = 0
			discount = 1.0
			self.EPSILON = self.EPSILON * self.EPSILON_DECAY

			############################################################

			while True:							
				############################################################
				# Pick the next action using epsilon greedy and and execute it
				############################################################				
				q_values = self.session.run(self.Q,feed_dict={self.x:[state]})
				action = q_values.argmax()
				# print q_values,action

				if np.random.uniform() <= self.EPSILON:					
					action = np.random.randint(self.output_size)

				############################################################
				# Step in the environment. Something like: 
				# next_state, reward, done, _ = self.env.step(action)
				############################################################

				nxt_state, reward, done, _ = self.env.step(action)
				episodic_reward += discount*reward
				discount *= self.DISCOUNT_FACTOR

				############################################################
				# Update the (limited) replay buffer. 
				#
				# Note : when the replay buffer is full, you'll need to 
				# remove an entry to accommodate a new one.
				############################################################

				self.replay_memory[curr_experience_num] = [state, action, reward, nxt_state, done]
				curr_experience_num += 1
				if len(self.replay_memory) > self.REPLAY_MEMORY_SIZE:
					del self.replay_memory[oldest_experience]
					oldest_experience = self.replay_memory.keys()[0]

				############################################################
				# Sample a random minibatch and perform Q-learning (fetch max Q at s')
				#
				# Remember, the target (r + gamma * max Q) is computed    
				# with the help of the target network.
				# Compute this target and pass it to the network for computing
				# and minimizing the loss with the current estimates
				#
				############################################################
				
				# print self.replay_memory[0]
				# print self.replay_memory[-1]
				# print " "
				transition_indices = np.random.choice(self.replay_memory.keys(),self.MINIBATCH_SIZE)

				# transitions = [self.replay_memory[i] for i in transition_indices]
				
				batch_states = np.reshape([self.replay_memory[i][0] for i in transition_indices], (self.MINIBATCH_SIZE,self.input_size))
				batch_actions = np.reshape([self.replay_memory[i][1] for i in transition_indices],(self.MINIBATCH_SIZE,1))
				batch_rewards = np.reshape([self.replay_memory[i][2] for i in transition_indices],(self.MINIBATCH_SIZE,1))
				batch_next_states = np.reshape([self.replay_memory[i][3] for i in transition_indices],(self.MINIBATCH_SIZE,self.input_size))
				batch_dones = np.reshape([self.replay_memory[i][4] for i in transition_indices],(self.MINIBATCH_SIZE,1))		


				# batch_next_q_vals = self.session.run(self.Q,feed_dict={self.x:batch_next_states})
				if total_steps % self.TARGET_UPDATE_FREQ == 1:
					saver.restore(self.session2,"~/dqn")
				batch_next_q_vals = self.session2.run(self.Q,feed_dict={self.x:batch_next_states})


				batch_max_q = batch_next_q_vals.max(axis=1)
				batch_max_q = np.reshape(batch_max_q,(len(batch_max_q),1))

				# If done , then target = reward, else target = reward + gamma*max_q
				batch_targets = batch_rewards + self.DISCOUNT_FACTOR * (batch_max_q * np.logical_not(batch_dones))
				# print batch_targets
				batch_targets = batch_targets.flatten()

				batch_actions = (batch_actions.flatten()).tolist()
				encoded_actions = np.zeros((len(batch_actions),self.output_size))
				encoded_actions[np.arange(len(batch_actions)),batch_actions] = 1
				
				# print batch_states,batch_actions,encoded_actions,batch_rewards,batch_next_states,batch_dones,batch_next_q_vals,batch_max_q, batch_targets
				# print batch_states,batch_actions, encoded_actions, batch_targets, batch_rewards, batch_max_q

				self.session.run(self.train_op,feed_dict={self.x:batch_states,self.action:encoded_actions,self.target:batch_targets})				
				############################################################
			  	# Update target weights. 
			  	#
			  	# Something along the lines of:
				# if total_steps % self.TARGET_UPDATE_FREQ == 0:
				# 	target_weights = self.session.run(self.weights)
				############################################################

				if total_steps % self.TARGET_UPDATE_FREQ == 0:
					saver.save(self.session, "~/dqn")

				############################################################
				# Break out of the loop if the episode ends
				#
				# Something like:
				# if done or (episode_length == self.MAX_STEPS):
				# 	break
				#
				############################################################
				episode_length += 1				
				total_steps += 1
				state = nxt_state
				if done or (episode_length == self.MAX_STEPS):
					break

			############################################################
			# Logging. 
			#
			# Very important. This is what gives an idea of how good the current
			# experiment is, and if one should terminate and re-run with new parameters
			# The earlier you learn how to read and visualize experiment logs quickly,
			# the faster you'll be able to prototype and learn.
			#
			# Use any debugging information you think you need.
			# For instance :


			if len(episode_window) < 100:				
				episode_window += [episode_length]
				avg_epsiode_length = sum(episode_window)/ len(episode_window)

			elif len(episode_window) == 100:
				episode_window += [episode_length]
				episode_window.pop(0)
				avg_epsiode_length = sum(episode_window)*1.0/len(episode_window)
			
			print("Training: Episode = %d, Length = %d, Window_avg = %f Global step = %d" % (episode, episode_length, avg_epsiode_length, total_steps))
			summary.value.add(tag="episode length", simple_value=episode_length)
			summary_writer.add_summary(summary, episode)

			avg_episode_length_list += [avg_epsiode_length]
			ep_count += 1

			if avg_epsiode_length > 195:
				print ("Done!")
				break

		return avg_episode_length_list,ep_count


	def fill_replay_memory(self):
		self.session.run(tf.global_variables_initializer())
		done = False
		state = self.env.reset()
		size = self.MINIBATCH_SIZE
		for i in range(size):
			q_values = self.session.run(self.Q,feed_dict={self.x:[state]})
			act = q_values.argmax()
			nxt_state,rew, done,_ = self.env.step(act)
			self.replay_memory[i] = [state,act,rew,nxt_state,done]
			if done:				
				state = self.env.reset()
			else:
				state = nxt_state
		print len(self.replay_memory)

	# def calc_target_q_vals(self,inputs):
	# 	new_sess = tf.Session()		
	# 	tf.train.Saver().restore(new_sess,"~/dqn")
	# 	target_q_vals = new_sess.run(self.Q,feed_dict={self.x:inputs})
	# 	new_sess.close()
	# 	return target_q_vals

	# Simple function to visually 'test' a policy
	def playPolicy(self):
		
		done = False
		steps = 0
		state = self.env.reset()
		
		# we assume the CartPole task to be solved if the pole remains upright for 200 steps
		while not done and steps < 200:
			self.env.render()
			q_vals = self.session.run(self.Q, feed_dict={self.x: [state]})
			action = q_vals.argmax()
			state, _, done, _ = self.env.step(action)
			steps += 1
		
		return steps


if __name__ == '__main__':

	batch_sizes = [20]

	for batch_size in batch_sizes:
		# Create and initialize the model
		dqn = DQN('CartPole-v0')
		dqn.MINIBATCH_SIZE = batch_size

		run_data = []
		# 1 Run per expt
		for run in range(1):		
			dqn.initialize_network()
			dqn.fill_replay_memory()

			print("\nStarting training...\n")
			ep_lengths, num_ep = dqn.train()
			print("\nFinished training...\nCheck out some demonstrations\n")
			plt.plot(ep_lengths)
			run_data += [ep_lengths]
		pickle.dump( run_data, open( "batch-"+str(batch_size), "wb" ) )
		plt.savefig("batch_"+str(batch_size))
		plt.clf()



	# Visualize the learned behaviour for a few episodes
	results = []
	for i in range(100):
		episode_length = dqn.playPolicy()
		print("Test steps = ", episode_length)
		results.append(episode_length)
	print("Mean steps = ", sum(results) / len(results))	

	print("\nFinished.")
	print("\nCiao, and hasta la vista...\n")