import gym, gym_test
import numpy as np
import time
import matplotlib.pyplot as plt
from itertools import product
import pickle

def choose_action(q,epsilon):
	# calc greedy action for q-learning updateS  --->epsilon-greedy ???
	greedy_action = np.argmax(q)

	# take behavior action which is epsilon greedy
	behavior_action = greedy_action
	epsilon = np.random.uniform()
	if epsilon <= 0.1:
		behavior_action = np.random.randint(env.action_space.n) #exploration

	return greedy_action, behavior_action

env = gym.make('test-v0')

problem = raw_input("Which Problem?")
if problem == "A":
	env.terminal_state = [0,11]
elif problem == "B":
	env.terminal_state = [2,9]
elif problem == "C":
	env.terminal_state = [6,7]
	env.wind_prob = 0.0
env.problem = problem

# Seed
np.random.seed(int(time.time()))

runs = 50
gamma = 0.9
alpha = 0.01
total_episodes = 20000

#Each value in both arrays wil avg of (runs) # of values  
avg_episode_returns = np.zeros(total_episodes)
avg_episode_lengths = np.zeros(total_episodes)
avg_Q = np.zeros((env.grid_side, env.grid_side,env.action_space.n))
Policy = np.random.randint(4,size=(env.grid_side,env.grid_side))

for j in range(runs):
	print "Run "+str(j)+" Time : "+str(time.time())
	
	#Algo Init
	Q = np.random.uniform(0,1,(env.grid_side, env.grid_side,env.action_space.n))
	#Terminal state val = 0
	for i in range(env.action_space.n):
		Q[env.terminal_state[0]][env.terminal_state[1]][i] = 0 

	# Env Init
	env.seed()
	y = env.reset()

	# Variables for 1 run consisting of total_episodes
	episode_lengths = []
	episode_returns = []
	avg_episode_length = 0.0
	avg_return = 0.0
	num_episodes = 0 
	time_step = 0

	gamma_cum = 1.0  # to maintain (gamma)^t
	episodic_return = 0

	# Initial epsilon-greedy action
	behavior_action = choose_action(Q[env.curr_state[0],env.curr_state[1]], 0.1)

	while num_episodes < total_episodes:		
		#store state and action
		curr_x,curr_y = env.curr_state[0],env.curr_state[1]	
		curr_action = behavior_action

		#Take step and observe r,s'
		obs = env.step(curr_action)
		
		# G_t += gamma^k * r
		episodic_return += obs[1]  #REMOVED GAMMA_CUM TO MAKE UNDISCOUNTED
		gamma_cum = gamma_cum * gamma
		
		greedy_action, behavior_action =  choose_action(Q[env.curr_state[0],env.curr_state[1]], 0.1) # 0.1 exploration

		#Sarsa Update
		Q[curr_x,curr_y,curr_action] += alpha*(obs[1] + gamma*Q[env.curr_state[0],env.curr_state[1],behavior_action] - Q[curr_x,curr_y,curr_action])

		if num_episodes == total_episodes - 1:
			env.render()

		#If Episode Done-->restart
		if obs[2] == True:
			y = env.reset()
			# print "============RESET================="
			# print env.curr_state
			num_episodes += 1

			avg_episode_length += (time_step  - avg_episode_length) / (num_episodes*1.0)
			avg_return += (episodic_return - avg_return)/(num_episodes*1.0)
			
			episode_lengths += [time_step]
			episode_returns += [episodic_return]

			if episodic_return > 10:
				print "WRONG"

			episodic_return = 0 
			gamma_cum = 1.0
			time_step = 0
		else:
			time_step += 1
		# env.render()
	print avg_episode_length,avg_return

	# Take avg of episode lengths and episode returns
	temp = np.divide(np.subtract(episode_lengths, avg_episode_lengths), j + 1)
	avg_episode_lengths  = np.add(avg_episode_lengths, temp)

	temp = np.divide(np.subtract(episode_returns, avg_episode_returns), j + 1)
	avg_episode_returns  = np.add(avg_episode_returns, temp)

	temp = np.divide(np.subtract(Q, avg_Q), j + 1)
	avg_Q = np.add(avg_Q,temp)


for (x,y) in product(np.arange(env.grid_side), np.arange(env.grid_side)):
	Policy[x,y] = np.argmax(avg_Q[x,y])


np.savetxt("Data/Sarsa/policy_sarsa_"+problem,Policy)
pickle.dump(avg_Q,open( "Data/Sarsa/Q-Val_sarsa_"+problem, "wb" ))
np.savetxt("Data/Sarsa/lengths_sarsa_"+problem,avg_episode_lengths)
np.savetxt("Data/Sarsa/returns_sarsa_"+problem,avg_episode_returns)