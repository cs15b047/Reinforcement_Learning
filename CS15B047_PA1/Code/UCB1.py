import numpy as np
from matplotlib import pyplot
from math import sqrt,log 
import pickle

# update Q+U values
def update(Q,N,t,c):
	ans = []
	for j in range(len(Q)):
		ans += [  ( Q[j] + c* sqrt(log(t+1)/N[j]) )  ]
	return ans


k=10
steps = 100
num_problems = 2000
bandit_problem = [] 

with open ('testbed', 'rb') as fp:
    bandit_problem = pickle.load(fp)

q_max = np.argmax(bandit_problem,axis=1)


average_rewards = []
optimal_actions = []
Avg_1000_steps = []
Error_list = []
c_list = [1]
# c_list = [1,2]

for c in c_list:
	avg_ucb1 = np.zeros(steps)
	opt_action_count = np.zeros(steps+1)
	error = 0
	average = 0

	for i in range(num_problems):
		# print (i)
		q = bandit_problem[i]
		Q = np.zeros(k)
		N = np.zeros(k)				
		temp_rewards = []
		temp = np.zeros(steps+1)

		for j in range(k):
			Q[j] = np.random.normal(q[j])
			temp_rewards += [Q[j]]
			N[j] += 1	
			if j == q_max[i]:
				temp[j+1] = 1
			else:
				temp[j+1] = 0

		Q_plus_U = update(Q,N,k,c)

		for t in range(k+1,steps+1):
			# deterministically pick actions
			action = np.argmax(Q_plus_U)
			if action == q_max[i]:
				temp[t] = 1
			else :
				temp[t] = 0
			reward = np.random.normal(q[action])
			N[action] += 1
			Q[action] += (reward - Q[action])/N[action]	
			temp_rewards += [reward]

			# update q+u values
			Q_plus_U = update(Q,N,t,c)

		for j in range(steps):
			avg_ucb1[j] += (temp_rewards[j] - avg_ucb1[j])/(i+1)

		average += (np.sum(temp_rewards)-average)/(i+1)
		opt_action_count = np.add(opt_action_count,temp)
		error += (q_max[i] != np.argmax(Q_plus_U))		

	print ("Error - c="+str(c)+" ",error/20.0)
	Error_list += [error/20.0]

	opt_action_count = np.multiply(np.divide(opt_action_count,2000.0),100)
	opt_action_count = opt_action_count[1:]
	average_rewards += [avg_ucb1]
	optimal_actions += [opt_action_count]
	Avg_1000_steps += [average]

with open('ucb1_q4', 'wb') as fp:
    pickle.dump([average_rewards,optimal_actions,c_list,Avg_1000_steps,Error_list], fp)
