import matplotlib.pyplot as plt
import pickle
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#Open data files

with open ('eps_greedy', 'rb') as fp:
    eps_greedy = pickle.load(fp)

with open ('softmax', 'rb') as fp:
    softmax = pickle.load(fp)

with open ('ucb1', 'rb') as fp:
    ucb1 = pickle.load(fp)

with open ('mea', 'rb') as fp:
    mea = pickle.load(fp)

#give ques no as input

ip = raw_input()

figure = plt.figure()
plt1 = figure.add_subplot(211)
plt2 = figure.add_subplot(212)


# Generate algo_q4 files first before trying to plot for q4, by changing file names in respective files
if ip == "q4":
	figure = plt.figure()
	ax1 = figure.add_subplot(211)
	ax = figure.add_subplot(212)
	with open ('eps_greedy_q4', 'rb') as fp:
	    eps_greedy_q4 = pickle.load(fp)

	with open ('softmax_q4', 'rb') as fp:
	    softmax_q4 = pickle.load(fp)

	with open ('ucb1_q4', 'rb') as fp:
	    ucb1_q4 = pickle.load(fp)

	with open ('mea_q4', 'rb') as fp:
	    mea_2 = pickle.load(fp)



if ip == "q1":
	plt.suptitle('Epsilon-greedy performance')
	for (data,eps) in zip(eps_greedy[0],eps_greedy[2]):	
		if eps == 1:
			plt1.plot(data,label = "eps=1/t")
		else:		
			plt1.plot(data,label = "eps="+str(eps))
	plt1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt1.set_xlabel('Steps')
	plt1.set_ylabel('Average Rewards')

	for (data,eps) in zip(eps_greedy[1],eps_greedy[2]):		
		if eps == 1:
			plt2.plot(data,label = "eps=1/t")
		else:		
			plt2.plot(data,label = "eps="+str(eps))
	plt2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt2.set_xlabel('Steps')
	plt2.set_ylabel('% Optimal Actions')

	figure = plt.figure()
	plt3 = figure.add_subplot(111)
	plt.suptitle('Epsilon - Greedy Suboptimal arm % ')
	plt3.plot(eps_greedy[2],eps_greedy[4])
	plt3.set_xlabel('Epsilon')
	plt3.set_ylabel('Error %')

elif ip == "q2":
	plt.suptitle('Softmax Performance')
	for (data,temp) in zip(softmax[0],softmax[2]):		
		if temp == 1:
			plt1.plot(data,label = "tmp=\n1/log(t)")
		else:	
			plt1.plot(data,label = "tmp="+str(temp))
	plt1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt1.set_xlabel('Steps')
	plt1.set_ylabel('Average Rewards')

	for (data,temp) in zip(softmax[1],softmax[2]):		
		if temp == 1:
			plt2.plot(data,label = "tmp=\n1/log(t)")	
		else:
			plt2.plot(data,label = "tmp="+str(temp))
	plt2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt2.set_xlabel('Steps')
	plt2.set_ylabel('% Optimal Actions')	

	figure = plt.figure()
	plt3 = figure.add_subplot(111)
	plt.suptitle('Softmax Suboptimal arm % ')
	plt3.plot(softmax[2],softmax[4])
	plt3.set_xlabel('C')
	plt3.set_ylabel('Error %')

elif ip == "q3":
	plt.suptitle('Performance Comparison')
	for (data,eps) in zip(eps_greedy[0],eps_greedy[2]):		
		plt1.plot(data,label = "eps="+str(eps))
	for (data,temp) in zip(softmax[0],softmax[2]):		
		plt1.plot(data,label = "tmp="+str(temp))
	for (data,c) in zip(ucb1[0],ucb1[2]):		
		plt1.plot(data,label= "c="+str(c))

	plt1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt1.set_xlabel('Steps')
	plt1.set_ylabel('Average Rewards')

	for (data,eps) in zip(eps_greedy[1],eps_greedy[2]):		
		plt2.plot(data,label = "eps="+str(eps))
	for (data,temp) in zip(softmax[1],softmax[2]):		
		plt2.plot(data,label = "tmp="+str(temp))
	for (data,c) in zip(ucb1[1],ucb1[2]):		
		plt2.plot(data,label= "c="+str(c))
	plt2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt2.set_xlabel('Steps')
	plt2.set_ylabel('% Optimal Actions')	

	figure = plt.figure()
	plt3 = figure.add_subplot(111)
	plt.suptitle('UCB1 Suboptimal arm % ')
	plt3.plot(ucb1[2],ucb1[4])
	plt3.set_xlabel('C')
	plt3.set_ylabel('Error %')

elif ip == "q4":
	# mea[2] = np.subtract(100.0, mea[2])
	# mea[2] = np.divide(mea[2],mea[3])

	# Z = np.asarray(mea[2]).reshape((len(mea[0]),len(mea[1]) ))

	# plt.suptitle('MEA Suboptimal arm %')

	# for i in range(len(mea[0])):
	# 	ax1.plot(mea[1],Z[i],label="mea_eps="+str(mea[0][i]))

	# ax1.legend()
	# ax1.set_xlabel('Delta')
	# ax1.set_ylabel('Error %')

	figure.suptitle("Performance Comparison between different algorithms")


	# Assumed that only one curve for each algorithm in comparison
	ax1.plot(softmax_q4[0][0],label="Softmax")
	ax1.plot(eps_greedy_q4[0][0],label="Epsilon-greedy")
	ax1.plot(ucb1_q4[0][0],label="UCB1")
	ax1.plot(mea_2[4],label="MEA")


	ax1.legend()
	ax1.set_xlabel('Steps')
	ax1.set_ylabel('Average Rewards')
	ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))


	ax.plot(softmax_q4[1][0],label="Softmax")
	ax.plot(eps_greedy_q4[1][0],label="Epsilon-greedy")
	ax.plot(ucb1_q4[1][0],label="UCB1")
	ax.plot(mea_2[5],label="MEA")


	ax.legend()
	ax.set_xlabel('Steps')
	ax.set_ylabel('Optimal Actions %')
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


plt.show()