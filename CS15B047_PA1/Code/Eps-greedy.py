import numpy as np
from matplotlib import pyplot
from math import sqrt,log 
import pickle

k=10
steps = 100
bandit_problem = [] 
num_problems = 2000

with open ('testbed', 'rb') as fp:
    bandit_problem = pickle.load(fp)

q_max = np.argmax(bandit_problem,axis=1)


opt_actions = []
average_rewards = []

epsilon_list = [0.1]
# epsilon_list = [0.1,0.3]
Avg_1000_steps = []

Error_list = []

flag = False

for epsilon in epsilon_list:
	# epsilon = 1 is considered as decaying epsilon case
	if epsilon == 1:
		flag = True
	error = 0 
	opt_action_count = np.zeros(steps+1)
	avg_eps_greedy = np.zeros(steps)
	average = 0
	for i in range(num_problems):
		# print (i)
		Q = np.zeros(k)
		N = np.zeros(k)
		q = bandit_problem[i]
		action = -1		
		temp = np.zeros(steps+1)
		temp_rewards = []

		for t in range(1,steps+1):
			if flag == True:
				epsilon = 1.0/t
			
			# Sample from distribution using idea of picking randomly uniformly from CDF
			rand = np.random.uniform()
			if(rand >= epsilon):
				action = np.argmax(Q) # greedy choice
			else:
				action = np.random.randint(k) # random exploration

			# Maintain counts for optimal action taken	
			if action == q_max[i]:
				temp[t] = 1
			else:
				temp[t] = 0

			# sample from rewrd distribution , variance =1 is implicit.
			reward = np.random.normal(q[action])
			temp_rewards += [reward]
			N[action]+= 1
			#incremental averages
			Q[action]+= (reward-Q[action])/N[action]
		average += (np.sum(temp_rewards) - average) / (i+1)
		for j in range(steps):
			avg_eps_greedy[j] += (temp_rewards[j]-avg_eps_greedy[j])/(i+1)

		opt_action_count = np.add(opt_action_count,temp)

		# calculate how many times final arm is suboptimal
		error += (q_max[i] != np.argmax(Q))
	print ("Error for "+str(epsilon)+" is ",error/20.0)
	Error_list += [error/20.0]
	Avg_1000_steps += [average]

	# print (opt_action_count)

	opt_action_count = np.multiply(np.divide(opt_action_count,2000.0),100)
	
	average_rewards += [avg_eps_greedy]
	opt_actions += [opt_action_count[1:]]

with open('eps_greedy', 'wb') as fp:
    pickle.dump([average_rewards,opt_actions,epsilon_list,Avg_1000_steps,Error_list], fp)