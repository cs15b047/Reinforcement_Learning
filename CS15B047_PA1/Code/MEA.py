import numpy as np
from matplotlib import pyplot
from math import sqrt,log,ceil
import pickle
import timeit

k=10
# steps = 10000
bandit_problem = [] 
num_problems = 500

with open ('testbed', 'rb') as fp:
    bandit_problem = pickle.load(fp)

q_max = np.argmax(bandit_problem,axis=1)

arr = []

Error_list = []
Time_list = []

# Epsilon_list = np.arange(0.1,0.3,0.1)
# Delta_list = np.arange(0.1,0.4,0.1)

Epsilon_list = [1.5]
Delta_list = [0.9]

for Epsilon in Epsilon_list:
	for Delta in Delta_list:
		print("Epsilon:",Epsilon)
		print ("Delta:",Delta)

		epsilon = Epsilon / 4
		delta = Delta / 2 

		avg_mea = np.zeros(3000000)
		opt_action = np.zeros(3000000)
		T = 0
		num_problems = num_problems
		start = timeit.default_timer()
		error = 0

		for i in range(num_problems):
			q = bandit_problem[i]
			Q = np.zeros(k)
			N = np.zeros(k)
			left_arms = np.arange(k)

			temp_rewards = []
			t = 0

			#initialization
			epsilon = Epsilon / 4
			delta = Delta / 2 

			temp_count = []
			x = 0
			while len(left_arms) > 1:
				x += 1
				#calculate pulls for each arm
				pulls = int(ceil((2/(epsilon*epsilon)) * log(3/delta)))

				# Pull arms roundwise /(arm by arm also possible)
				for p in range(pulls):
					for a in left_arms:	
						t+=1
						N[a] += 1
						reward = np.random.normal(q[a])
						temp_rewards += [reward]
						Q[a] +=  (reward - Q[a])/N[a]
						
						if a == q_max[i]:
							temp_count += [1]
						else :
							temp_count += [0] 
				left_Q = []
				for a in left_arms:
					left_Q +=[Q[a]]

				# calculate median
				med = np.median(left_Q)
				new_left = []

				# eliminate arms
				for a in left_arms:
					if Q[a] >= med:
						new_left += [a]
				
				# Update parmeters
				left_arms = new_left
				epsilon = 3*epsilon/4
				delta = delta/2

			error += (-q[left_arms[0]] + bandit_problem[i][q_max[i]])!= 0

			opt_action = np.add(opt_action[:len(temp_count)],temp_count)
			T = t
			print (T)
			print (i,left_arms)

			avg_mea = avg_mea[:len(temp_rewards)]

			for j in range(len(temp_rewards)):
				avg_mea[j] += (temp_rewards[j] - avg_mea[j])/(i+1)

		end = timeit.default_timer()

		print (end - start)
		exec_time = end-start
		Time_list += [exec_time]
		print ("Accuracy :",error*100.0/(num_problems))

		Error_list += [error*100.0/(num_problems)]

		opt_action = np.multiply(np.divide(opt_action,num_problems),100)
		opt_action = opt_action[1:]

# pyplot.plot(avg_mea)
# pyplot.show()

# pyplot.plot(opt_action)
# pyplot.show()

with open('mea_q4', 'wb') as fp:
    pickle.dump([Epsilon_list,Delta_list,Error_list,Time_list,avg_mea,opt_action], fp)

print (Error_list)
