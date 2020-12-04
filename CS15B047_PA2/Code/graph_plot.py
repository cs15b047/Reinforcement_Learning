from mpl_toolkits.mplot3d import Axes3D
from matplotlib.axes import Axes

import pygame
import numpy as np
import matplotlib.pyplot as plt
import pickle
# Q = np.random.uniform(0,1,(2,2,3))
# x = [0,0]
# print Q
# print np.argmax(Q[x[0],x[1]])

def policy_conv(policy):
	print policy
	for i in np.arange(len(policy)):
		y = 0.8 - i*1.0/20.0
		for j in np.arange(len(policy[i])):
			x = 0.1 + j*1.0/20.0
			if policy[i][j] == 0 :
				ax.arrow(x,y,0,-0.05)
			if policy[i][j] == 1 :
				ax.arrow(x,y,0.05,0)
			if policy[i][j] == 2 :
				ax.arrow(x,y,0,0.05)
			if policy[i][j] == 3 :
				ax.arrow(x,y,-0.05,0)	

ax = plt.axes()
# ax.arrow(0,0,.2,.4)

for problem in ["A","B","C"]:
	for what in ["returns_25","lengths_25"]:

		suffix1 = "_sarsa_lambda_"+problem
		suffix2 = "_sarsa_"+problem
		suffix3 = "_Q_"+problem
		loc_prefix1 = "Q1/Data/Sarsa-lambda/"
		loc_prefix2 = "Q1/Data/Sarsa/"
		loc_prefix3 = "Q1/Data/Q/"
		
		print_policy = []

		print "Sarsa Policy"
		policy = np.loadtxt(loc_prefix2+"policy"+suffix2)
		ax = plt.axes()
		policy_conv(policy)

		plt.show()

		print "Q Policy"
		policy = np.loadtxt(loc_prefix3+"policy"+suffix3)
		ax = plt.axes()
		policy_conv(policy)
		plt.show()
		# Q = pickle.load(open( "Q-Val", "rb" ))

		# print policy
		# print Q	

		# change acc to _Q,_sarsa,_sarsa_lambda
		# avg_episode_lengths = np.loadtxt(loc_prefix+"lengths"+suffix)
		# plt.plot(avg_episode_lengths)

		print "Sarsa-lambda Policy"
		for Lambda in [0,0.3,0.5,0.9,0.99,1.0]:
			avg_episode_returns = np.loadtxt(loc_prefix1+what+suffix1+"_"+str(Lambda))
			# plt.plot(avg_episode_returns,label= "Lambda:"+str(Lambda))									
			policy = np.loadtxt(loc_prefix1+"policy"+suffix1+"_"+str(Lambda))
			ax = plt.axes()
			policy_conv(policy)	
			plt.show()
		# avg_episode_returns = np.loadtxt(loc_prefix2+what+suffix2)
		# plt.plot(avg_episode_returns,label="Sarsa")
		
		# avg_episode_returns = np.loadtxt(loc_prefix3+what+suffix3)
		# plt.plot(avg_episode_returns,label="Q")
		
		# plt.xlabel("Episodes")
		# plt.ylabel(what)
		# plt.legend()
		# plt.show()		


# a = [1,2,4]
# for x in a[::-1]:
# 	print x
# a = np.arange(-40,0,1)
# print a

# screen = pygame.display.set_mode((640, 480))
# running = 1
 
# while running:
# 	event = pygame.event.poll()
# 	if event.type == pygame.QUIT:
# 		running = 0

# 	screen.fill((255, 255, 255))
# 	# for (x,y) in product(np.arange(12),np.arange(12)):			
# 	pygame.draw.line(screen, (0, 0, 255), (0, 0), (639, 200))
# 	pygame.draw.aaline(screen, (0, 0, 255), (639, 0), (0, 479))
# 	pygame.display.flip()

# loc_prefix = "Q2/chakra/"

# ret1 = np.loadtxt(loc_prefix+"Distance_1.0_500_0.64")
# plt.plot(ret1)
# ret1 = np.loadtxt(loc_prefix+"Distance_0.6_500_0.64")
# plt.plot(ret1)
# ret1 = np.loadtxt(loc_prefix+"Distance_0.3_500_0.64")
# plt.plot(ret1)
# plt.show()


# # Trajectories Q2
loc_prefix = "Q2/chakra/"
ret1 = pickle.load(open( loc_prefix+"Trajectory_1.0_50_0.1", "rb" ))
ret2 = np.loadtxt(loc_prefix+"Values_1.0_50_0.1")

# print ret2[0]

x = []
y = []
z = []
# print len(ret1[0])

ret1 = np.asarray(ret1)
for i in np.arange(ret1.shape[0]/10):	
	# print ret2[i]		
	for (p,r) in zip(ret1[i],ret2[i]):
		if x != [] and y != [] and np.linalg.norm(np.asarray([x[-1]-p[1],y[-1]-p[2]])) > 0.2:
			pass
		else:
			x += [p[1]]
			y += [p[2]]
			z += [r]

figure = plt.figure()
ax = figure.add_subplot(111,projection='3d')
ax.scatter(x,y,z,c='r',marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

# for gamma in [1.0,0.6,0.3]:
# dist = np.loadtxt(loc_prefix+"Tuning/Distance_1.0_500_0.1")
# plt.plot(dist,label="1.0_500_0.1")	
# dist = np.loadtxt(loc_prefix+"Distance_1.0_50_0.1")
# plt.plot(dist,label="1.0_500-50_0.1")
# plt.xlabel("Iterations")
# plt.ylabel("Distance from origin")
# plt.legend()
# plt.show()




