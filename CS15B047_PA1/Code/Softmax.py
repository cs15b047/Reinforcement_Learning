import numpy as np
from matplotlib import pyplot
from math import sqrt,log 
import pickle

# Calculate prob given Q and temp
def calc_prob(Q,temperature):
	temp = np.divide(Q,temperature)
	temp = np.exp(temp) / np.sum(np.exp(temp), axis=0)
	return temp


k=10
steps = 100
bandit_problem = [] 
num_problems = 2000

with open ('testbed', 'rb') as fp:
    bandit_problem = pickle.load(fp)

q_max = np.argmax(bandit_problem,axis=1)
temps = [0.3]

average_rewards = []
opt_action = []
Avg_1000_steps = []
Error_list = []

flag = False

for temperature in temps:
	#temperature = 1 --> cooling of 1/log(t) here
	if temperature == 1:
		flag = True

	opt_action_count = np.zeros(steps+1)
	avg_softmax = np.zeros(steps)
	average = 0
	error = 0
	
	for i in range(num_problems):
		# print (i)
		Q = np.zeros(k)
		N = np.zeros(k)
		q = bandit_problem[i]
		action = -1
		temp = np.zeros(steps+1)
		
		#Start with equal 1/k probabilities
		prob = np.ones(k)
		prob = np.divide(prob,k)
		cum_prob = np.cumsum(prob)
		temp_rewards = []
		
		for t in range(1,steps+1):
			if flag == True:
				temperature = 1.0/log(t+1)		
			rand = np.random.uniform()

			# Sample from CDF and check whose CDF interval captures value, leading to sampling from PDF
			for a in np.arange(k):				
				if(rand <= cum_prob[a]):
					action = a
					break			

			if action == q_max[i]:
				temp[t] = 1
			else:
				temp[t] = 0

			reward = np.random.normal(q[action])
			temp_rewards += [reward]
			N[action]+= 1
			Q[action]+= (reward-Q[action])/N[action]
			
			#update probabilities
			prob = calc_prob(Q,temperature)
			cum_prob = np.cumsum(prob)			
		#suboptimality of final arm
		error += (q_max[i] != np.argmax(Q))

		average += (np.sum(temp_rewards) - average)/(i+1)
			
		for j in range(steps):
			avg_softmax[j] += (temp_rewards[j]-avg_softmax[j])/(i+1)
		opt_action_count = np.add(opt_action_count,temp)

	print ("Error :",error/20.0)
	Error_list += [error/20.0]
	opt_action_count = np.multiply(np.divide(opt_action_count,2000.0),100)	
	opt_action += [opt_action_count[1:]]
	average_rewards += [avg_softmax]
	Avg_1000_steps += [average]


with open('softmax_q4', 'wb') as fp:
    pickle.dump([average_rewards,opt_action,temps,Avg_1000_steps,Error_list], fp)